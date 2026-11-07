# Benchmarks

Every run of `bench.py --save` adds one row. Greedy decoding. tok/s is the total for all users.

| date | commit | model | dtype | machine | users | prompt tokens | new tokens | tok/s |
|---|---|---|---|---|---|---|---|---|
| 2026-11-07 | 1f93314 | Qwen/Qwen3-1.7B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 1 | 5 | 20 | 5.9 |
| 2026-11-07 | 1f93314 | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 1 | 5 | 20 | 10.6 |
| 2026-11-07 | d834bad | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 1 | 5 | 20 | 15.8 |
| 2026-11-07 | d834bad | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 1 | 5 | 200 | 12.4 |
| 2026-11-07 | be27b5c | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 1 | 5 | 20 | 17.4 |
| 2026-11-07 | be27b5c | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 4 | 5 | 20 | 46.6 |
| 2026-11-07 | be27b5c | Qwen/Qwen3-0.6B | bf16 | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 threads | 16 | 5 | 20 | 99.5 |
