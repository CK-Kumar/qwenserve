import argparse
import os

import uvicorn

# The qwenserve command:
#   qwenserve
#   qwenserve --model Qwen/Qwen3-1.7B --port 8001


def main():
    parser = argparse.ArgumentParser(description="OpenAI-style server for Qwen3 models")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to let other machines in")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.environ["QWENSERVE_MODEL"] = args.model  # server.py reads it when it loads
    uvicorn.run("qwenserve.server:app", host=args.host, port=args.port)
