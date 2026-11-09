import os
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from huggingface_hub.errors import RepositoryNotFoundError
from tokenizers import Tokenizer

from qwenserve.chat import chat_prompt
from qwenserve.config import load_config
from qwenserve.loader import load_weights
from qwenserve.model import generate_batch

# An OpenAI-style chat server.
#   uv run uvicorn qwenserve.server:app --port 8000
#   QWENSERVE_MODEL=Qwen/Qwen3-1.7B uv run uvicorn qwenserve.server:app --port 8000

# uvicorn starts this file itself, so the model comes from an environment variable, not a flag
MODEL = os.environ.get("QWENSERVE_MODEL", "Qwen/Qwen3-0.6B")

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

app = FastAPI()


@app.get("/v1/models")
async def models():
    # clients ask this first, to know which model name to send
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "qwenserve"}]}


@app.post("/v1/chat/completions")
async def chat_completions(body: dict):
    # body is the request JSON as a plain dict: {"messages": [...], "max_tokens": 100}
    if body.get("model", MODEL) != MODEL:
        # don't answer with another model than the one asked for, same error as vLLM
        message = f"The model `{body['model']}` does not exist."
        return JSONResponse({"error": {"message": message, "type": "NotFoundError", "code": 404}}, status_code=404)
    ids = tok.encode(chat_prompt(body["messages"])).ids
    # async def + slow work here = the whole server waits until this answer is done,
    # so only one request at a time. The background engine loop will fix that.
    text = generate_batch(
        [ids],
        w,
        c,
        body.get("max_tokens", 256),
        # the client's choice, else the model's own defaults, like vLLM
        temperature=body.get("temperature", c.temperature),
        top_k=body.get("top_k", c.top_k),
        top_p=body.get("top_p", c.top_p),
        seed=body.get("seed"),
    )[0]
    new = text[len(ids) :]
    answer = tok.decode(new, skip_special_tokens=True)
    # ended on its own, or cut off at max_tokens
    finish_reason = "stop" if new[-1] in c.eos_token_ids else "length"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": len(ids), "completion_tokens": len(new), "total_tokens": len(ids) + len(new)},
    }
