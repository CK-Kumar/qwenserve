# Benchmarks

Every run of `bench.py --save` adds one row. Greedy decoding, batch of 1.

| date | commit | model | dtype | machine | prompt tokens | new tokens | tok/s |
|---|---|---|---|---|---|---|---|
| 2026-11-07 | 1f93314 | Qwen/Qwen3-1.7B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 5 | 20 | 5.9 |
| 2026-11-07 | 1f93314 | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 5 | 20 | 10.6 |
