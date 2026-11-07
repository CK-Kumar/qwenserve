import torch

from qwenserve.layers import attention, mlp, rms_norm


def forward(ids, w, c, caches=None, counts=None):
    # ids are the new tokens of every user, one after the other.
    # counts[b] of them belong to user b, and caches[b] has b's older tokens.
    if caches is None:  # just one user with nothing cached
        caches = [[None] * c.num_hidden_layers]
        counts = [len(ids)]

    # Each user counts positions from their own start.
    positions = []
    for cache, n in zip(caches, counts):
        # how many tokens are already cached, all layers have the same count
        if cache[0] is None:
            start = 0
        else:
            k, v = cache[0]
            start = k.shape[0]
        positions.append(torch.arange(start, start + n))
    positions = torch.cat(positions)

    x = w["model.embed_tokens.weight"][ids]  # (T, hidden_size)
    for i in range(c.num_hidden_layers):
        p = f"model.layers.{i}."
        # Each block adds its result to x instead of replacing it (the "residual").
        x = x + attention(rms_norm(x, w[p + "input_layernorm.weight"], c.rms_norm_eps), w, c, i, positions, caches, counts)
        x = x + mlp(rms_norm(x, w[p + "post_attention_layernorm.weight"], c.rms_norm_eps), w, i)
    x = rms_norm(x, w["model.norm.weight"], c.rms_norm_eps)
    return x @ w["lm_head.weight"].T


def generate(ids, w, c, max_new_tokens):
    # one user is just a batch of 1
    return generate_batch([ids], w, c, max_new_tokens)[0]


def generate_batch(prompts, w, c, max_new_tokens):
    # prompts is a list of token id lists, one per user.
    # Greedy: always take the highest scoring token.
    # For now every user gets max_new_tokens, nobody stops early.
    # TODO (0.3): let one user stop at eos while the others go on
    texts = [list(ids) for ids in prompts]  # full text per user, to return at the end
    caches = [[None] * c.num_hidden_layers for _ in prompts]
    new = [list(ids) for ids in prompts]  # first step sends every whole prompt (prefill)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            counts = [len(n) for n in new]
            flat = [t for n in new for t in n]  # all users' new tokens in one list
            logits = forward(torch.tensor(flat), w, c, caches, counts)

            # each user's next token comes from their last row
            new = []
            row = 0
            for b, n in enumerate(counts):
                row += n
                next_id = int(logits[row - 1].argmax())
                texts[b].append(next_id)
                new.append([next_id])  # after prefill, 1 token per user
    return texts
