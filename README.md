# qwenserve

I wanted to know what really happens inside an inference server like vLLM when
you send it a prompt. So I'm building a small one myself, from scratch in
PyTorch, for Qwen models.

The code is meant to be read. Each idea gets its own small file: the KV cache,
batching, paged memory, the server. If you're learning how inference works,
you can follow it step by step, the same way I did.

## Supported models

Runs the Qwen3 dense models:

| model | download (bf16) |
|---|---|
| Qwen3-0.6B | 1.5 GB |
| Qwen3-1.7B | 4.1 GB |
| Qwen3-4B, Qwen3-4B-Instruct-2507 | 8.0 GB |
| Qwen3-8B | 16.4 GB |
| Qwen3-14B | 29.5 GB |
| Qwen3-32B | 65.5 GB |

## Versions

| version | models | what's new | speed (laptop CPU, bf16) |
|---|---|---|---|
| 0.1 | Qwen3 dense | Loads any Qwen3 dense model and generates text. Same tokens as Hugging Face (tested on 0.6B). No KV cache yet. | 0.6B: 10.6 tok/s, 1.7B: 5.9 tok/s |
| 0.2 | Qwen3 dense | KV cache, several users in one batch, paged KV cache (pages of 16 tokens), one attention call for all users in decode. | 0.6B: 17.1 tok/s for 1 user, 249.5 tok/s total for 64 users |
| 0.3 | Qwen3 dense | Sampling (temperature, top_k, top_p, seed). Each user stops at its own end and gives its pages back. A waiting line: users wait for a free seat and for room in the KV pool, which is sized from free RAM. A stats line every few seconds, like vLLM's. | 0.6B: 17.4 tok/s for 1 user, 297.1 tok/s total for 64 users |
| 0.4 | Qwen3 dense | OpenAI-style server: `/v1/models`, `/v1/chat/completions`, `/v1/completions`, with streaming. Requests that come in at different times share the same steps. Tested with the official openai client. | 0.6B: 16.0 tok/s for 1 user, 256.2 tok/s total for 64 users. Through the server, 16 users at once: 106 tok/s total, first text after 0.1–0.6 s |

## Install

```
uv tool install git+https://github.com/CK-Kumar/qwenserve
```
or with pip:
```
pip install git+https://github.com/CK-Kumar/qwenserve
```

On Linux, PyTorch installs its big GPU build by default. For CPU only, install it first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`

## Run the server

```
qwenserve
qwenserve --help
```
(inside a clone of this repo: `uv run qwenserve`)

Any OpenAI client works with it:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
r = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "What is a CPU?"}],
)
print(r.choices[0].message.content)
```

Another model or port: `qwenserve --model Qwen/Qwen3-1.7B --port 8001`.
At start it takes half of the free RAM for the KV cache.

Measure it with many users at once: `uv run python bench_server.py --users 16`

## How each version is built

1. Add one idea (KV cache, batching, ...).
2. Check it still gives the same tokens as Hugging Face: `uv run pytest tests`
3. Measure it: `uv run python bench.py --model Qwen/Qwen3-0.6B --save`
   Every run goes into [BENCHMARKS.md](BENCHMARKS.md).
4. Set `version` in `pyproject.toml` to match.
5. Tag it, so you can read any step on its own: `git checkout v0.1`

## Roadmap

More Qwen models I want to add later:

| model | what it needs |
|---|---|
| Qwen3-30B-A3B, Qwen3-235B-A22B (MoE) | expert routing in the MLP |
| Qwen3.5, 0.8B to 9B | linear attention layers, which keep a different kind of cache |
| Qwen3.6 | the same as Qwen3.5, and a bigger GPU |
