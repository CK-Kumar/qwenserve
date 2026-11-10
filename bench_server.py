import argparse
import statistics
import threading
import time

from openai import OpenAI

# Measures the server the way users see it: many streaming requests at once.
# Start the server first, in another terminal: uv run qwenserve
#   uv run python bench_server.py --users 16

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:8000/v1")
parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
parser.add_argument("--users", type=int, default=16, help="how many requests at the same time")
parser.add_argument("--tokens", type=int, default=50, help="max new tokens per request")
args = parser.parse_args()

client = OpenAI(base_url=args.url, api_key="none")
first = []  # seconds until each user saw their first text
counts = []  # new tokens each user got


def one_user():
    start = time.time()
    stream = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "Explain what a CPU does."}],
        max_tokens=args.tokens,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},  # last chunk has the token count
    )
    seen = False
    for chunk in stream:
        if not seen and chunk.choices and chunk.choices[0].delta.content:
            first.append(time.time() - start)
            seen = True
        if chunk.usage:
            counts.append(chunk.usage.completion_tokens)


start = time.time()
threads = [threading.Thread(target=one_user) for _ in range(args.users)]
for t in threads:
    t.start()
for t in threads:
    t.join()
secs = time.time() - start

print(f"users       {args.users}, {args.tokens} max new tokens each")
print(f"total       {sum(counts)} tokens in {secs:.1f} s  →  {sum(counts) / secs:.1f} tok/s")
print(f"first text  median {statistics.median(first):.2f} s, slowest {max(first):.2f} s")
