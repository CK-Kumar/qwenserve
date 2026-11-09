import os
import queue
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from huggingface_hub.errors import RepositoryNotFoundError
from tokenizers import Tokenizer

from qwenserve.chat import chat_prompt
from qwenserve.config import load_config
from qwenserve.engine import start
from qwenserve.kv_cache import PAGE, page_bytes, pages_from_memory
from qwenserve.loader import load_weights
from qwenserve.model import make_engine

# An OpenAI-style chat server.
#   uv run uvicorn qwenserve.server:app --port 8000
#   QWENSERVE_MODEL=Qwen/Qwen3-1.7B uv run uvicorn qwenserve.server:app --port 8000

# uvicorn starts this file itself, so the model comes from an environment variable, not a flag
MODEL = os.environ.get("QWENSERVE_MODEL", "Qwen/Qwen3-0.6B")
MAX_RUNNING = 32  # seats: users answered at the same time, the rest wait

# Load once when the server starts, not per request (takes seconds).
# Config first: it checks the model is one we can run. On a bad model, stop with
# one clear line instead of a 40-line traceback.
try:
    c = load_config(MODEL)
except ValueError as e:
    raise SystemExit(f"qwenserve can't run this model: {e}.\nSupported: Qwen3 dense, 0.6B to 32B (see README).")
except RepositoryNotFoundError:
    raise SystemExit(f"qwenserve can't find {MODEL} on Hugging Face. Check the name.")
tok = Tokenizer.from_pretrained(MODEL)
w = load_weights(MODEL, c)

# One engine for the whole server, its cook runs in the background from now on.
dtype = w["model.embed_tokens.weight"].dtype
num_pages = pages_from_memory(c, dtype)  # after the weights, they take RAM too
print(f"KV pool: {num_pages} pages ({num_pages * page_bytes(c, dtype) / 1e9:.1f} GB), {num_pages * PAGE} tokens, {MAX_RUNNING} seats")
engine = make_engine(w, c, num_pages, MAX_RUNNING, log=True)
start(engine)

app = FastAPI()


@app.get("/v1/models")
async def models():
    # clients ask this first, to know which model name to send
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "qwenserve"}]}


@app.post("/v1/chat/completions")
def chat_completions(body: dict):
    # plain def: FastAPI runs each request in its own small thread, so it can just
    # wait at its mailbox while the others are served
    # body is the request JSON as a plain dict: {"messages": [...], "max_tokens": 100}
    if body.get("model", MODEL) != MODEL:
        # don't answer with another model than the one asked for, same error as vLLM
        message = f"The model `{body['model']}` does not exist."
        return JSONResponse({"error": {"message": message, "type": "NotFoundError", "code": 404}}, status_code=404)
    ids = tok.encode(chat_prompt(body["messages"])).ids
    mailbox = queue.Queue()
    engine["rail"].put(
        {
            "ids": ids,
            "max_new_tokens": body.get("max_tokens", 256),
            # the client's choice, else the model's own defaults, like vLLM
            "temperature": body.get("temperature", c.temperature),
            "top_k": body.get("top_k", c.top_k),
            "top_p": body.get("top_p", c.top_p),
            "seed": body.get("seed"),
            "mailbox": mailbox,
        }
    )
    # The cook sends one (token, finish_reason) per step. Collect them all,
    # finish_reason is None until the last one.
    new = []
    finish_reason = None
    while finish_reason is None:
        msg = mailbox.get()  # sleeps until the cook sends something
        if isinstance(msg, ValueError):  # it can never fit in the pool
            return JSONResponse({"error": {"message": str(msg), "type": "BadRequestError", "code": 400}}, status_code=400)
        token, finish_reason = msg
        new.append(token)
    answer = tok.decode(new, skip_special_tokens=True)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": len(ids), "completion_tokens": len(new), "total_tokens": len(ids) + len(new)},
    }
