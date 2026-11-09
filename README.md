# qwenserve

I wanted to know what really happens inside an inference server like vLLM when
you send it a prompt. So I'm building a small one myself, from scratch in
PyTorch, for Qwen models.

The code is meant to be read. Each idea gets its own small file: the KV cache,
batching, paged memory, prefix caching. If you're learning how inference works,
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

## How each version is built

1. Add one idea (KV cache, batching, ...).
2. Check it still gives the same tokens as Hugging Face: `uv run pytest tests`
3. Measure it: `uv run python bench.py --model Qwen/Qwen3-0.6B --save`
   Every run goes into [BENCHMARKS.md](BENCHMARKS.md).
4. Tag it, so you can read any step on its own: `git checkout v0.1`

## Roadmap

More Qwen models I want to add later:

| model | what it needs |
|---|---|
| Qwen3-30B-A3B, Qwen3-235B-A22B (MoE) | expert routing in the MLP |
| Qwen3.5, 0.8B to 9B | linear attention layers, which keep a different kind of cache |
| Qwen3.6 | the same as Qwen3.5, and a bigger GPU |
