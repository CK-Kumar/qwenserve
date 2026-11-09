import argparse
import os
import sys

import uvicorn

# The qwenserve command:
#   qwenserve
#   qwenserve --model Qwen/Qwen3-1.7B --port 8001


class Parser(argparse.ArgumentParser):
    # same short error as argparse, plus where to look
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, f"qwenserve: error: {message}\nRun 'qwenserve --help' to see all options.\n")


def main():
    parser = Parser(
        description="OpenAI-style server for Qwen3 models, on CPU.",
        epilog="""examples:
  qwenserve
  qwenserve --model Qwen/Qwen3-1.7B --port 8001

then call http://localhost:8000/v1/chat/completions with any OpenAI client.
At start it takes half of the free RAM for the KV cache.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,  # keep the line breaks above
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Hugging Face name of a Qwen3 dense model, 0.6B to 32B (default: %(default)s)")
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on, 0.0.0.0 lets other machines in (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8000, help="port to listen on (default: %(default)s)")
    args = parser.parse_args()
    os.environ["QWENSERVE_MODEL"] = args.model  # server.py reads it when it loads
    uvicorn.run("qwenserve.server:app", host=args.host, port=args.port)
