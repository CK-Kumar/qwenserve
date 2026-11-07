import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwenserve.config import load_config
from qwenserve.loader import load_weights
from qwenserve.model import forward, generate

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
