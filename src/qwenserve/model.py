import torch

from qwenserve.layers import attention, mlp, rms_norm


def forward(ids, w, c):
    # ids: (T,) token ids  →  logits: (T, vocab_size), a score for every possible next token
    positions = torch.arange(len(ids))
    x = w["model.embed_tokens.weight"][ids]  # (T, hidden_size)
    for i in range(c.num_hidden_layers):
        p = f"model.layers.{i}."
        # Each block adds its result to x instead of replacing it (the "residual").
        x = x + attention(rms_norm(x, w[p + "input_layernorm.weight"], c.rms_norm_eps), w, c, i, positions)
        x = x + mlp(rms_norm(x, w[p + "post_attention_layernorm.weight"], c.rms_norm_eps), w, i)
    x = rms_norm(x, w["model.norm.weight"], c.rms_norm_eps)
    return x @ w["lm_head.weight"].T


def generate(ids, w, c, max_new_tokens, eos_id=None):
    # Greedy: always take the highest scoring token.
    # No KV cache yet, so every step runs the whole text again. Slow, but simple.
    # TODO (0.2): keep k and v from earlier steps instead of recomputing them
    ids = list(ids)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = forward(torch.tensor(ids), w, c)
            next_id = int(logits[-1].argmax())
            ids.append(next_id)
            if next_id == eos_id:
                break
    return ids
