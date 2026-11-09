import json
import os
import queue
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from huggingface_hub.errors import RepositoryNotFoundError
from tokenizers import Tokenizer

from qwenserve.chat import chat_prompt, new_text
from qwenserve.config import load_config
from qwenserve.engine import start
from qwenserve.kv_cache import PAGE, page_bytes, pages_from_memory
from qwenserve.loader import load_weights
from qwenserve.model import make_engine, pages_needed

# OpenAI-style server: /v1/models, /v1/chat/completions, /v1/completions.
#   uv run uvicorn qwenserve.server:app --port 8000
#   QWENSERVE_MODEL=Qwen/Qwen3-1.7B uv run uvicorn qwenserve.server:app --port 8000

# uvicorn imports this file, so the model comes from an env var, not a flag
MODEL = os.environ.get("QWENSERVE_MODEL", "Qwen/Qwen3-0.6B")
MAX_RUNNING = 32  # max users at once, the rest wait

# Load once at start. Config first, it checks we support the model.
# On a bad model, print one line instead of a long traceback.
try:
    c = load_config(MODEL)
except ValueError as e:
    raise SystemExit(f"qwenserve can't run this model: {e}.\nSupported: Qwen3 dense, 0.6B to 32B (see README).")
except RepositoryNotFoundError:
    raise SystemExit(f"qwenserve can't find {MODEL} on Hugging Face. Check the name.")
tok = Tokenizer.from_pretrained(MODEL)
w = load_weights(MODEL, c)

# One engine for the whole server, running in the background.
dtype = w["model.embed_tokens.weight"].dtype
num_pages = pages_from_memory(c, dtype)  # after the weights, they take RAM too
print(f"KV pool: {num_pages} pages ({num_pages * page_bytes(c, dtype) / 1e9:.1f} GB), {num_pages * PAGE} tokens, {MAX_RUNNING} seats")
engine = make_engine(w, c, num_pages, MAX_RUNNING, log=True)
start(engine)

app = FastAPI()


@app.get("/v1/models")
async def models():
    # clients call this first to get the model name
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "qwenserve"}]}


@app.post("/v1/chat/completions")
def chat_completions(body: dict):
    # plain def, not async: FastAPI runs it in its own thread, so it can wait on the mailbox
    # body is the request JSON, e.g. {"messages": [...], "max_tokens": 100}
    ids = tok.encode(chat_prompt(body["messages"])).ids
    mailbox, err = submit(body, ids, default_max_tokens=256)
    if err:
        return err
    rid = f"chatcmpl-{uuid.uuid4().hex}"

    if body.get("stream"):

        def events():
            yield sse(chat_chunk(rid, {"role": "assistant", "content": ""}, None))  # first chunk only has the role
            for piece, finish_reason, n in pieces(mailbox):
                if piece or finish_reason:
                    yield sse(chat_chunk(rid, {"content": piece} if piece else {}, finish_reason))
            if body.get("stream_options", {}).get("include_usage"):
                yield sse({"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": MODEL, "choices": [], "usage": usage(ids, n)})
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    new, finish_reason = collect(mailbox)
    return {
        "id": rid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": tok.decode(new, skip_special_tokens=True)}, "finish_reason": finish_reason}],
        "usage": usage(ids, len(new)),
    }


@app.post("/v1/completions")
def completions(body: dict):
    # no chat template, the model just continues the prompt
    # body e.g. {"prompt": "The capital of France is", "max_tokens": 5}
    ids = tok.encode(body["prompt"]).ids
    mailbox, err = submit(body, ids, default_max_tokens=16)  # 16 is the OpenAI default
    if err:
        return err
    rid = f"cmpl-{uuid.uuid4().hex}"

    if body.get("stream"):

        def events():
            for piece, finish_reason, n in pieces(mailbox):
                if piece or finish_reason:
                    yield sse(completion_chunk(rid, piece, finish_reason))
            if body.get("stream_options", {}).get("include_usage"):
                yield sse({"id": rid, "object": "text_completion", "created": int(time.time()), "model": MODEL, "choices": [], "usage": usage(ids, n)})
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    new, finish_reason = collect(mailbox)
    chunk = completion_chunk(rid, tok.decode(new, skip_special_tokens=True), finish_reason)
    chunk["usage"] = usage(ids, len(new))
    return chunk


def submit(body, ids, default_max_tokens):
    # Checks the request and sends it to the engine.
    # Returns (mailbox, None) or (None, error).
    if body.get("model", MODEL) != MODEL:
        # wrong model name, same error as vLLM
        return None, error(f"The model `{body['model']}` does not exist.", "NotFoundError", 404)
    max_new_tokens = body.get("max_tokens", default_max_tokens)
    # check here: once a stream starts it's too late to return a 400
    if pages_needed(ids, max_new_tokens) > num_pages:
        return None, error("this request needs more pages than the whole pool, it would wait forever", "BadRequestError", 400)
    mailbox = queue.Queue()
    engine["rail"].put(
        {
            "ids": ids,
            "max_new_tokens": max_new_tokens,
            # request value, else the model's defaults (like vLLM)
            "temperature": body.get("temperature", c.temperature),
            "top_k": body.get("top_k", c.top_k),
            "top_p": body.get("top_p", c.top_p),
            "seed": body.get("seed"),
            "mailbox": mailbox,
        }
    )
    return mailbox, None


def tokens(mailbox):
    # one (token, finish_reason) per step, finish_reason is None until the end
    finish_reason = None
    while finish_reason is None:
        msg = mailbox.get()  # waits for the engine
        if isinstance(msg, Exception):
            raise msg
        token, finish_reason = msg
        yield token, finish_reason


def collect(mailbox):
    # no streaming: wait for the whole answer
    new = []
    for token, finish_reason in tokens(mailbox):
        new.append(token)
    return new, finish_reason


def pieces(mailbox):
    # streaming: new text each step, "" while a character is incomplete
    new, sent = [], ""
    for token, finish_reason in tokens(mailbox):
        new.append(token)
        piece = new_text(tok, new, sent, done=finish_reason is not None)
        sent += piece
        yield piece, finish_reason, len(new)


def chat_chunk(rid, delta, finish_reason):
    return {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def completion_chunk(rid, text, finish_reason):
    return {
        "id": rid,
        "object": "text_completion",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason}],
    }


def usage(ids, n):
    return {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n}


def sse(obj):
    # one SSE message: "data: ..." and a blank line
    return f"data: {json.dumps(obj)}\n\n"


def error(message, kind, code):
    return JSONResponse({"error": {"message": message, "type": kind, "code": code}}, status_code=code)
