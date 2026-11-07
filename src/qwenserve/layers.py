import torch
import torch.nn.functional as F

# The building blocks of one Qwen3 layer. Plain functions: each one takes the
# weights dict `w`, the config `c` and the layer number `i`.
# x is always (tokens, hidden_size), one row per token.


def rms_norm(x, weight, eps):
    # Scale every row so its numbers have size about 1, then apply the learned weight.
    # Done in float32 like Hugging Face: in bf16 the squares lose too much precision.
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return weight * x.to(dtype)


def rope(x, positions, theta):
    # x: (tokens, heads, head_dim). Turns pairs of numbers by an angle that grows
    # with the token's position, so attention can tell near from far.
    dim = x.shape[-1]
    speeds = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))  # (64,)
    angles = positions[:, None].float() * speeds[None, :]  # (tokens, 64)
    angles = torch.cat([angles, angles], dim=-1)[:, None, :]  # (tokens, 1, 128)
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    # Qwen pairs number j with number j + 64, not with its neighbour.
    first, second = x[..., : dim // 2], x[..., dim // 2 :]
    turned = torch.cat([-second, first], dim=-1)
    return x * cos + turned * sin


def attention(x, w, c, i, positions):
    p = f"model.layers.{i}.self_attn."
    T, H, KV, D = x.shape[0], c.num_attention_heads, c.num_key_value_heads, c.head_dim

    q = (x @ w[p + "q_proj.weight"].T).view(T, H, D)
    k = (x @ w[p + "k_proj.weight"].T).view(T, KV, D)
    v = (x @ w[p + "v_proj.weight"].T).view(T, KV, D)

    # Qwen3 normalizes each head of q and k before RoPE (Qwen2 didn't).
    q = rope(rms_norm(q, w[p + "q_norm.weight"], c.rms_norm_eps), positions, c.rope_theta)
    k = rope(rms_norm(k, w[p + "k_norm.weight"], c.rms_norm_eps), positions, c.rope_theta)

    # Fewer k/v heads than q heads, so each k/v head is shared by a group of q heads.
    # 0.6B: 16 / 8 = 2.  8B: 32 / 8 = 4.  32B: 64 / 8 = 8.
    group = H // KV
    q = q.transpose(0, 1)  # (H, T, D): heads first
    k = k.repeat_interleave(group, dim=1).transpose(0, 1)
    v = v.repeat_interleave(group, dim=1).transpose(0, 1)

    scores = q @ k.transpose(-2, -1) / D**0.5  # (H, T, T)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))  # a token can't look ahead
    out = scores.softmax(dim=-1, dtype=torch.float32).to(v.dtype) @ v  # (H, T, D)

    out = out.transpose(0, 1).reshape(T, H * D)  # glue the heads back together
    return out @ w[p + "o_proj.weight"].T  # (T, hidden_size)


def mlp(x, w, i):
    p = f"model.layers.{i}.mlp."
    gate = F.silu(x @ w[p + "gate_proj.weight"].T)  # (T, intermediate_size)
    up = x @ w[p + "up_proj.weight"].T
    return (gate * up) @ w[p + "down_proj.weight"].T  # (T, hidden_size)
