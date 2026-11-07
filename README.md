# qwenserve

I wanted to know what really happens inside an inference server like vLLM when
you send it a prompt. So I'm building a small one myself, from scratch in
PyTorch, for Qwen models.

The code is meant to be read. Each idea gets its own small file: the KV cache,
batching, paged memory, prefix caching. If you're learning how inference works,
you can follow it step by step, the same way I did.

## Supported models

Version 0.1 runs the Qwen3 dense models:

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
