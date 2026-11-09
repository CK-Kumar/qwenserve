import argparse
import platform
import subprocess
import time
from datetime import date

import torch
from tokenizers import Tokenizer

from qwenserve.config import load_config
from qwenserve.loader import load_weights
from qwenserve.model import generate_batch

# Measures how fast qwenserve generates, on this machine.
#   uv run python bench.py --model Qwen/Qwen3-0.6B
#   uv run python bench.py --model Qwen/Qwen3-1.7B --tokens 50 --save
#   uv run python bench.py --users 4

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
parser.add_argument("--prompt", default="The capital of France is")
parser.add_argument("--tokens", type=int, default=20, help="how many new tokens to generate")
parser.add_argument("--users", type=int, default=1, help="how many users in one batch, all with the same prompt")
parser.add_argument("--max-running", type=int, default=None, help="seats: at most this many users at once, the rest wait")
parser.add_argument("--log", action="store_true", help="print one line per step")
parser.add_argument("--save", action="store_true", help="add the result to BENCHMARKS.md")
args = parser.parse_args()


def cpu_name():
    # On Linux the real CPU name is in /proc/cpuinfo; elsewhere fall back to e.g. "arm64".
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":")[1].strip()
    except OSError:
        pass
    return platform.machine()


tok = Tokenizer.from_pretrained(args.model)
c = load_config(args.model)

start = time.time()
w = load_weights(args.model, c)
load_s = time.time() - start

ids = tok.encode(args.prompt).ids
prompts = [ids] * args.users
generate_batch(prompts, w, c, 1)  # warm up: the very first run is always slower

start = time.time()
out = generate_batch(prompts, w, c, args.tokens, max_running=args.max_running, log=args.log)
gen_s = time.time() - start
tok_s = args.users * args.tokens / gen_s  # all users together

print(f"model     {args.model}")
print(f"users     {args.users}")
print(f"prompt    {len(ids)} tokens, generated {args.tokens} each")
print(f"load      {load_s:.1f} s")
print(f"generate  {gen_s:.1f} s  →  {tok_s:.1f} tok/s")
print(f"text      {tok.decode(out[0])!r}")

if args.save:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    machine = cpu_name()
    row = (
        f"| {date.today()} | {commit} | {args.model} | bf16 | {machine}, {torch.get_num_threads()} threads "
        f"| {args.users} | {len(ids)} | {args.tokens} | {tok_s:.1f} |\n"
    )
    with open("BENCHMARKS.md", "a") as f:
        f.write(row)
    print("saved to BENCHMARKS.md")
