from pydantic import BaseModel

# What a request body can hold. FastAPI fills these from the JSON and
# rejects wrong types. Ranges (top_p in (0, 1] ...) are checked in server.check().
# Fields we don't support (n, stop, user ...) are ignored.


class Request(BaseModel):
    # None means "not sent": the server uses its own default.
    # model can't default to MODEL here, importing server.py would load the weights.
    model: str | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None
    stream: bool = False
    stream_options: dict | None = None  # {"include_usage": true}


class CompletionRequest(Request):
    prompt: str
    max_tokens: int = 16  # the OpenAI default


class ChatRequest(Request):
    messages: list[dict]  # [{"role": "user", "content": "hi"}]
    max_tokens: int = 256
