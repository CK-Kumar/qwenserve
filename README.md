# qwenserve

I wanted to know what really happens inside an inference server like vLLM when
you send it a prompt. So I'm building a small one myself, from scratch in
PyTorch, for Qwen models.

The code is meant to be read. Each idea gets its own small file: the KV cache,
batching, paged memory, prefix caching. If you're learning how inference works,
you can follow it step by step, the same way I did.

## Supported models

Version 0.1 runs the Qwen3 dense models:

| model | size in bf16 |
|---|---|
| Qwen3-0.6B | 1.2 GB |
| Qwen3-1.7B | 3.4 GB |
| Qwen3-4B, Qwen3-4B-Instruct-2507 | 8 GB |
| Qwen3-8B | 16 GB |
| Qwen3-14B | 28 GB |
| Qwen3-32B | 64 GB |

## Versions

| version | models | what's new |
|---|---|---|
| 0.1 (in progress) | Qwen3 dense | |

## Roadmap

More Qwen models I want to add later:

| model | what it needs |
|---|---|
| Qwen3-30B-A3B, Qwen3-235B-A22B (MoE) | expert routing in the MLP |
| Qwen3.5, 0.8B to 9B | linear attention layers, which keep a different kind of cache |
| Qwen3.6 | the same as Qwen3.5, and a bigger GPU |
