import socket
import subprocess
import sys
import time
import urllib.request
from urllib.error import URLError

import openai
import pytest

# Runs the real server and checks it with the official openai client.

MODEL = "Qwen/Qwen3-0.6B"
CHAT = [{"role": "user", "content": "What is the capital of France? Just the name."}]


@pytest.fixture(scope="module")
def client():
    with socket.socket() as s:  # ask the system for a free port
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "qwenserve.server:app", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}/v1"
    for _ in range(120):  # loading the model takes a few seconds
        try:
            urllib.request.urlopen(f"{url}/models")
            break
        except URLError:
            time.sleep(1)
    yield openai.OpenAI(base_url=url, api_key="none")  # no API keys here
    server.terminate()
    server.wait()


def test_models(client):
    assert [m.id for m in client.models.list()] == [MODEL]


def test_chat(client):
    r = client.chat.completions.create(model=MODEL, messages=CHAT, temperature=0)
    assert r.choices[0].message.content == "Paris"
    assert r.choices[0].finish_reason == "stop"
    assert r.usage.completion_tokens == 2  # "Paris" and <|im_end|>


def test_chat_stream_same_as_whole(client):
    whole = client.chat.completions.create(model=MODEL, messages=CHAT, temperature=0, max_tokens=30)
    chunks = list(client.chat.completions.create(model=MODEL, messages=CHAT, temperature=0, max_tokens=30, stream=True))
    text = "".join(ch.choices[0].delta.content or "" for ch in chunks)
    assert text == whole.choices[0].message.content
    assert chunks[-1].choices[0].finish_reason == whole.choices[0].finish_reason


def test_completions_stream_same_as_whole(client):
    args = dict(model=MODEL, prompt="The capital of France is", max_tokens=5, temperature=0)
    whole = client.completions.create(**args)
    text = "".join(ch.choices[0].text for ch in client.completions.create(**args, stream=True))
    assert whole.choices[0].text.startswith(" Paris")
    assert text == whole.choices[0].text
    assert whole.choices[0].finish_reason == "length"


def test_wrong_model(client):
    with pytest.raises(openai.NotFoundError):
        client.chat.completions.create(model="gpt-4", messages=CHAT)


def test_too_big(client):
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(model=MODEL, messages=CHAT, max_tokens=100_000_000)
