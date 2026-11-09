import pytest
from transformers import AutoTokenizer

from qwenserve.chat import chat_prompt

CHATS = [
    [{"role": "user", "content": "What is a KV cache?"}],
    [{"role": "system", "content": "Answer in one sentence."}, {"role": "user", "content": "What is a KV cache?"}],
    [  # an old answer with thinking in it, the thinking must be dropped
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "<think>\nSimple sum.\n</think>\n\n4"},
        {"role": "user", "content": "And 3 + 3?"},
    ],
]


@pytest.mark.parametrize("messages", CHATS)
@pytest.mark.parametrize("thinking", [False, True])
def test_same_as_hf(messages, thinking):
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    hf = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    assert chat_prompt(messages, thinking) == hf
