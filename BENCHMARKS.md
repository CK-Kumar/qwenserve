# Benchmarks

Every run of `bench.py --save` adds one row. Greedy decoding. tok/s is the total for all users.
From 2026-11-10 on, tok/s is the median of 5 runs. Before that it was one run, and runs differ by about ±10%.

| date | commit | model | dtype | machine | users | prompt tokens | new tokens | tok/s |
|---|---|---|---|---|---|---|---|---|
| 2026-11-07 | 1f93314 | Qwen/Qwen3-1.7B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 1 | 5 | 20 | 5.9 |
| 2026-11-07 | 1f93314 | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 1 | 5 | 20 | 10.6 |
| 2026-11-07 | d834bad | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 1 | 5 | 20 | 15.8 |
| 2026-11-07 | d834bad | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 1 | 5 | 200 | 12.4 |
| 2026-11-07 | be27b5c | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 1 | 5 | 20 | 17.4 |
| 2026-11-07 | be27b5c | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 4 | 5 | 20 | 46.6 |
| 2026-11-07 | be27b5c | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 16 | 5 | 20 | 99.5 |
| 2026-11-07 | 2657e4c | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 1 | 5 | 20 | 17.1 |
| 2026-11-07 | 2657e4c | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 16 | 5 | 20 | 141.5 |
| 2026-11-07 | 2657e4c | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 64 | 5 | 20 | 249.5 |
| 2026-11-07 | 2657e4c | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 1 | 5 | 200 | 15.3 |
| 2026-11-09 | c9d7acc | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 1 | 5 | 20 | 17.4 |
| 2026-11-09 | c9d7acc | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 16 | 5 | 20 | 166.9 |
| 2026-11-09 | c9d7acc | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 64 | 5 | 20 | 297.1 |
| 2026-11-09 | b1a68d2 | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 1 | 5 | 20 | 16.0 |
| 2026-11-09 | b1a68d2 | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 16 | 5 | 20 | 146.0 |
| 2026-11-09 | b1a68d2 | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 64 | 5 | 20 | 256.2 |
| 2026-11-10 | e272b89 | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 1 | 5 | 20 | 18.6 |
| 2026-11-10 | e272b89 | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 16 | 5 | 20 | 145.1 |
| 2026-11-10 | e272b89 | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 64 | 5 | 20 | 250.4 |
