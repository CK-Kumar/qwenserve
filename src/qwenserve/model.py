import torch

from qwenserve.layers import attention, mlp, rms_norm


def forward(ids, w, c, cache=None):
    # ids are only the new tokens, cache has k and v of the older ones
    if cache is None:
        cache = [None] * c.num_hidden_layers
    # how many tokens are already cached, all layers have the same count
    if cache[0] is None:
        start = 0
    else:
        k, v = cache[0]
        start = k.shape[0]
    positions = torch.arange(start, start + len(ids))
    x = w["model.embed_tokens.weight"][ids]  # (T, hidden_size)
    for i in range(c.num_hidden_layers):
        p = f"model.layers.{i}."
        # Each block adds its result to x instead of replacing it (the "residual").
        x = x + attention(rms_norm(x, w[p + "input_layernorm.weight"], c.rms_norm_eps), w, c, i, positions, cache)
        x = x + mlp(rms_norm(x, w[p + "post_attention_layernorm.weight"], c.rms_norm_eps), w, i)
    x = rms_norm(x, w["model.norm.weight"], c.rms_norm_eps)
    return x @ w["lm_head.weight"].T


def generate(ids, w, c, max_new_tokens, eos_id=None):
    # Greedy: always take the highest scoring token.
    ids = list(ids)
    cache = [None] * c.num_hidden_layers
    new = ids  # first step sends the whole prompt (prefill)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = forward(torch.tensor(new), w, c, cache)
            next_id = int(logits[-1].argmax())
            ids.append(next_id)  # full text, to return at the end
            if next_id == eos_id:
                break
            new = [next_id]  # after that only the newest token, the rest is in the cache
    return ids
