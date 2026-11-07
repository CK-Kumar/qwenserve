import torch
import torch.nn.functional as F

# One Qwen3 layer, as plain functions.
# w = weights dict, c = config, i = layer number.
# x is (tokens, hidden_size), one row per token.


def rms_norm(x, weight, eps):
    # in float32 like HF does, bf16 loses too much when squaring
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return weight * x.to(dtype)


def rope(x, positions, theta):
    # Rotates q and k by an angle that depends on the position,
    # this is how attention knows the word order.
    dim = x.shape[-1]
    speeds = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))  # (64,)
    angles = positions[:, None].float() * speeds[None, :]  # (tokens, 64)
    angles = torch.cat([angles, angles], dim=-1)[:, None, :]  # (tokens, 1, 128)
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    # careful: Qwen pairs j with j + 64, not j with j + 1
    first, second = x[..., : dim // 2], x[..., dim // 2 :]
    turned = torch.cat([-second, first], dim=-1)
    return x * cos + turned * sin


def attention(x, w, c, i, positions, cache):
    # x has only the new tokens, older ones are already in cache[i]
    p = f"model.layers.{i}.self_attn."
    T, H, KV, D = x.shape[0], c.num_attention_heads, c.num_key_value_heads, c.head_dim

    q = (x @ w[p + "q_proj.weight"].T).view(T, H, D)
    k = (x @ w[p + "k_proj.weight"].T).view(T, KV, D)
    v = (x @ w[p + "v_proj.weight"].T).view(T, KV, D)

    # new in Qwen3, Qwen2 didn't norm q and k
    q = rope(rms_norm(q, w[p + "q_norm.weight"], c.rms_norm_eps), positions, c.rope_theta)
    k = rope(rms_norm(k, w[p + "k_norm.weight"], c.rms_norm_eps), positions, c.rope_theta)

    # KV cache: old k and v don't change, so keep them and add the new rows.
    # We store 8 kv heads here, before the repeat below, so half the memory.
    if cache[i] is not None:
        old_k, old_v = cache[i]
        k = torch.cat([old_k, k], dim=0)
        v = torch.cat([old_v, v], dim=0)
    cache[i] = (k, v)
    S = k.shape[0]  # all tokens so far
    start = S - T  # tokens that were already cached

    # GQA: each kv head is shared by H // KV query heads (2 for 0.6B, 4 for 8B)
    group = H // KV
    q = q.transpose(0, 1)  # (H, T, D)
    k = k.repeat_interleave(group, dim=1).transpose(0, 1)
    v = v.repeat_interleave(group, dim=1).transpose(0, 1)

    scores = q @ k.transpose(-2, -1) / D**0.5  # (H, T, S)

    # Hide keys that come after the token.
    # In prefill this is a triangle, in decode nothing gets hidden.
    q_pos = torch.arange(start, start + T)[:, None]
    k_pos = torch.arange(S)[None, :]
    future = k_pos > q_pos  # (T, S)
    scores = scores.masked_fill(future, float("-inf"))
    out = scores.softmax(dim=-1, dtype=torch.float32).to(v.dtype) @ v  # (H, T, D)

    out = out.transpose(0, 1).reshape(T, H * D)  # heads back side by side
    return out @ w[p + "o_proj.weight"].T


def mlp(x, w, i):
    p = f"model.layers.{i}.mlp."
    gate = F.silu(x @ w[p + "gate_proj.weight"].T)
    up = x @ w[p + "up_proj.weight"].T
    return (gate * up) @ w[p + "down_proj.weight"].T
