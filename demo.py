from tokenizers import Tokenizer

from qwenserve.config import load_config
from qwenserve.kv_cache import PAGE, page_bytes, pages_from_memory
from qwenserve.loader import load_weights
from qwenserve.model import generate_batch

# Watch continuous batching: 4 users, 2 seats, each one stops at its own time.
#   uv run python demo.py

MODEL = "Qwen/Qwen3-0.6B"
QUESTIONS = [
    "Say hi in one word.",
    "Name one film with Meryl Streep. Just the title.",
    "In one sentence, who is Emma Stone?",
    "Name three films with Zendaya.",
]


def chat(question):
    # Qwen3's chat format. The empty think block turns thinking off.
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


tok = Tokenizer.from_pretrained(MODEL)
c = load_config(MODEL)
w = load_weights(MODEL, c)
dtype = w["model.embed_tokens.weight"].dtype
num_pages = pages_from_memory(c, dtype)  # after the weights, they take RAM too
print(f"KV pool: {num_pages} pages ({num_pages * page_bytes(c, dtype) / 1e9:.1f} GB), {num_pages * PAGE} tokens")
prompts = [tok.encode(chat(q)).ids for q in QUESTIONS]
out = generate_batch(prompts, w, c, 60, max_running=2, num_pages=num_pages, log=True, stats_every=1)

print()
for i, (q, ids, text) in enumerate(zip(QUESTIONS, prompts, out)):
    answer = tok.decode(text[len(ids) :], skip_special_tokens=True)
    print(f"user {i}: {q}\n   → {answer}")
