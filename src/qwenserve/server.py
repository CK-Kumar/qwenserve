import time
import uuid

from fastapi import FastAPI
from tokenizers import Tokenizer

from qwenserve.chat import chat_prompt
from qwenserve.config import load_config
from qwenserve.loader import load_weights
from qwenserve.model import generate_batch

# An OpenAI-style chat server.
#   uv run uvicorn qwenserve.server:app --port 8000

MODEL = "Qwen/Qwen3-0.6B"

# Load once when the server starts, not per request (takes seconds).
tok = Tokenizer.from_pretrained(MODEL)
c = load_config(MODEL)
w = load_weights(MODEL, c)

app = FastAPI()


@app.post("/v1/chat/completions")
async def chat_completions(body: dict):
    # body is the request JSON as a plain dict: {"messages": [...], "max_tokens": 100}
    ids = tok.encode(chat_prompt(body["messages"])).ids
    # async def + slow work here = the whole server waits until this answer is done,
    # so only one request at a time. The background engine loop will fix that.
    text = generate_batch([ids], w, c, body.get("max_tokens", 256))[0]
    answer = tok.decode(text[len(ids) :], skip_special_tokens=True)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}}],
    }
