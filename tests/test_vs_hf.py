import queue

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwenserve.config import load_config
from qwenserve.engine import start
from qwenserve.loader import load_weights
from qwenserve.model import add_request, forward, generate, generate_batch, make_engine, step

# Our engine is only right if it gives what Hugging Face gives, on the same weights.
#
# We compare in float32, not bf16. bf16 rounds scores to steps of 0.125 near 16,
# so two tokens that are close can come out tied, and then the order of the math
# decides. Real case: after "Paris. The capital of", Italy beats France by 0.024
# in float32 in both engines. In bf16 HF gets a tie and picks France, qwenserve picks Italy.
# Neither is a bug, but it would make this test fail for the wrong reason.
NAME = "Qwen/Qwen3-0.6B"
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "The first person to walk on the Moon was",
]


@pytest.fixture(scope="module")
def setup():
    tok = AutoTokenizer.from_pretrained(NAME)
    c = load_config(NAME)
    w = {k: v.float() for k, v in load_weights(NAME, c).items()}
    hf = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32)
    return tok, c, w, hf


def test_logits_match(setup):
    tok, c, w, hf = setup
    ids = tok(PROMPTS[0], return_tensors="pt").input_ids
    with torch.no_grad():
        expected = hf(ids).logits[0]
        actual = forward(ids[0], w, c)
    assert (actual - expected).abs().max() < 1e-3


@pytest.mark.parametrize("prompt", PROMPTS)
def test_same_tokens(setup, prompt):
    tok, c, w, hf = setup
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        expected = hf.generate(ids, max_new_tokens=20, do_sample=False)[0].tolist()
    actual = generate(ids[0].tolist(), w, c, 20)
    assert actual == expected, f"\nexpected: {tok.decode(expected)!r}\nactual:   {tok.decode(actual)!r}"


# Users of different lengths in one batch must get the same text as each alone.
def test_batch_same_as_alone(setup):
    tok, c, w, hf = setup
    prompts = [tok(p).input_ids for p in PROMPTS]
    expected = [generate(ids, w, c, 20) for ids in prompts]
    actual = generate_batch(prompts, w, c, 20)
    assert actual == expected


# Same seed must give the same text, and keeping only the best token is just greedy.
def test_sampling(setup):
    tok, c, w, hf = setup
    ids = tok(PROMPTS[0]).input_ids
    a = generate(ids, w, c, 10, temperature=0.7, seed=1)
    b = generate(ids, w, c, 10, temperature=0.7, seed=1)
    assert a == b
    assert generate(ids, w, c, 10, temperature=0.7, top_k=1) == generate(ids, w, c, 10)


# A user who says <|im_end|> stops there, the other one goes on to the max.
def test_stop_at_eos(setup):
    tok, c, w, hf = setup
    chat = "<|im_start|>user\nSay hi in one word.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prompts = [tok(chat).input_ids, tok(PROMPTS[0]).input_ids]
    short, long = generate_batch(prompts, w, c, 25)
    assert short[-1] in c.eos_token_ids
    assert len(short) - len(prompts[0]) < 25
    assert len(long) - len(prompts[1]) == 25


def test_waiting_for_a_seat(setup):
    # one seat for three users must give the same texts as everyone at once
    tok, c, w, hf = setup
    prompts = [tok(p).input_ids for p in PROMPTS[:3]]
    assert generate_batch(prompts, w, c, 10, max_running=1) == generate_batch(prompts, w, c, 10)


def test_waiting_for_pages(setup):
    # a pool of 2 pages fits only some users at a time, texts must not change
    tok, c, w, hf = setup
    prompts = [tok(p).input_ids for p in PROMPTS[:3]]
    assert generate_batch(prompts, w, c, 10, num_pages=2) == generate_batch(prompts, w, c, 10)


def test_own_sampling_per_request(setup):
    # in one batch A is greedy and B samples, each must match itself alone
    tok, c, w, hf = setup
    a, b = tok(PROMPTS[0]).input_ids, tok(PROMPTS[1]).input_ids
    engine = make_engine(w, c, num_pages=10, max_running=2)
    ra = add_request(engine, a, 10)
    rb = add_request(engine, b, 10, temperature=0.7, seed=1)
    while engine["running"] or engine["waiting"]:
        step(engine)
    assert ra["text"] == generate(a, w, c, 10)
    assert rb["text"] == generate(b, w, c, 10, temperature=0.7, seed=1)


def test_engine_survives_a_crash(setup):
    # top_p=0 crashes sample() (the server's check() stops it, here we skip that)
    tok, c, w, hf = setup
    engine = make_engine(w, c, num_pages=10, max_running=2)
    start(engine)

    def order(top_p):
        mailbox = queue.Queue()
        ids = tok(PROMPTS[0]).input_ids
        engine["rail"].put({"ids": ids, "max_new_tokens": 3, "temperature": 0.7, "top_k": 0, "top_p": top_p, "seed": 1, "mailbox": mailbox})
        return mailbox

    bad = order(top_p=0)
    assert isinstance(bad.get(timeout=30), RuntimeError)  # the error comes back, no hang
    good = order(top_p=1.0)
    assert [good.get(timeout=30)[1] for _ in range(3)] == [None, None, "length"]  # 3 tokens, then done
    assert engine["reserved"] == 0 and len(engine["pool"]["free"]) == 10  # all pages back
