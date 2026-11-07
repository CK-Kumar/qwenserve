import torch

# Paged KV cache: one big pool made once, cut into pages of 16 tokens.
# A user only gets a new page when their last one is full.

PAGE = 16  # tokens per page, same as vLLM's default


def make_pool(c, num_pages, dtype):
    # one seat holds one token's k (or v) for one layer: (kv heads, head_dim)
    shape = (c.num_hidden_layers, num_pages, PAGE, c.num_key_value_heads, c.head_dim)
    return {
        "k": torch.zeros(shape, dtype=dtype),
        "v": torch.zeros(shape, dtype=dtype),
        "free": list(range(num_pages)),
    }


def new_user():
    # table: the user's pages in order. length: tokens stored so far.
    return {"table": [], "length": 0}


def grow(pool, user, n):
    # Make room for n more tokens and return where each one goes.
    # Done once per step, all layers use the same seats.
    pages, slots = [], []
    for _ in range(n):
        if user["length"] % PAGE == 0:  # last page is full (or there is none yet)
            if not pool["free"]:
                raise RuntimeError("out of KV pages, make the pool bigger")
            user["table"].append(pool["free"].pop(0))
        pages.append(user["table"][user["length"] // PAGE])
        slots.append(user["length"] % PAGE)
        user["length"] += 1
    return torch.tensor(pages), torch.tensor(slots)
