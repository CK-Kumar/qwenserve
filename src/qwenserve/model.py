import torch

from qwenserve.kv_cache import PAGE, free, grow, make_pool, new_user
from qwenserve.layers import attention, mlp, rms_norm
from qwenserve.sampling import sample


def forward(ids, w, c, pool=None, users=None, counts=None):
    # ids are the new tokens of every user, one after the other.
    # counts[b] of them belong to user b, and their older tokens are in the pool.
    if pool is None:  # just one user with nothing cached, like in the tests
        pool = make_pool(c, len(ids) // PAGE + 1, w["model.embed_tokens.weight"].dtype)
        users, counts = [new_user()], [len(ids)]

    # Each user counts positions from their own start, and gets seats for the
    # new tokens. Once per step, all layers write to the same seats.
    positions, seats = [], []
    for user, n in zip(users, counts):
        start = user["length"]  # tokens already cached
        seats.append(grow(pool, user, n))
        positions.append(torch.arange(start, start + n))
    positions = torch.cat(positions)

    x = w["model.embed_tokens.weight"][ids]  # (T, hidden_size)
    for i in range(c.num_hidden_layers):
        p = f"model.layers.{i}."
        # Each block adds its result to x instead of replacing it (the "residual").
        x = x + attention(rms_norm(x, w[p + "input_layernorm.weight"], c.rms_norm_eps), w, c, i, positions, pool, users, seats)
        x = x + mlp(rms_norm(x, w[p + "post_attention_layernorm.weight"], c.rms_norm_eps), w, i)
    x = rms_norm(x, w["model.norm.weight"], c.rms_norm_eps)
    return x @ w["lm_head.weight"].T


def generate(ids, w, c, max_new_tokens, temperature=0, top_k=0, top_p=1.0, seed=None):
    # one user is just a batch of 1
    return generate_batch([ids], w, c, max_new_tokens, temperature, top_k, top_p, seed)[0]


def generate_batch(prompts, w, c, max_new_tokens, temperature=0, top_k=0, top_p=1.0, seed=None):
    # prompts is a list of token id lists, one per user.
    # temperature 0 is greedy. Same seed gives the same text.
    # A user stops at eos or after max_new_tokens, the others go on.
    texts = [list(ids) for ids in prompts]  # full text per user, to return at the end
    # Sized so everyone fits, for now. A real server sizes it from free memory
    # and makes users wait when it's full (0.3).
    num_pages = sum((len(ids) + max_new_tokens) // PAGE + 1 for ids in prompts)
    pool = make_pool(c, num_pages, w["model.embed_tokens.weight"].dtype)
    users = [new_user() for _ in prompts]
    new = [list(ids) for ids in prompts]  # first step sends every whole prompt (prefill)
    running = list(range(len(prompts)))  # users still talking, by their number b
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    with torch.no_grad():
        while running:
            counts = [len(new[b]) for b in running]
            flat = [t for b in running for t in new[b]]  # running users' new tokens in one list
            logits = forward(torch.tensor(flat), w, c, pool, [users[b] for b in running], counts)

            # each user's next token comes from their last row
            still = []
            row = 0
            for b, n in zip(running, counts):
                row += n
                next_id = sample(logits[row - 1], temperature, generator, top_k, top_p)
                texts[b].append(next_id)
                new[b] = [next_id]  # after prefill, 1 token per user
                if next_id in c.eos_token_ids or len(texts[b]) - len(prompts[b]) == max_new_tokens:
                    free(pool, users[b])
                else:
                    still.append(b)
            running = still
    return texts
