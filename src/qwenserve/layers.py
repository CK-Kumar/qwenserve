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


def attention(x, w, c, i, positions, pool, users, seats):
    # x has the new tokens of every user stacked: user 0's rows, then user 1's...
    # seats[b] says where user b's new tokens go in the pool.
    p = f"model.layers.{i}.self_attn."
    N, H, KV, D = x.shape[0], c.num_attention_heads, c.num_key_value_heads, c.head_dim

    # all users together, so each weight is read once
    q = (x @ w[p + "q_proj.weight"].T).view(N, H, D)
    k = (x @ w[p + "k_proj.weight"].T).view(N, KV, D)
    v = (x @ w[p + "v_proj.weight"].T).view(N, KV, D)

    # new in Qwen3, Qwen2 didn't norm q and k
    q = rope(rms_norm(q, w[p + "q_norm.weight"], c.rms_norm_eps), positions, c.rope_theta)
    k = rope(rms_norm(k, w[p + "k_norm.weight"], c.rms_norm_eps), positions, c.rope_theta)

    # Users with 1 new token all go in one call. Users reading a prompt go one by
    # one, each only looks at their own words.
    out = torch.empty(N, H * D, dtype=q.dtype)
    rows, dec_users, dec_seats = [], [], []  # the 1-token users, collected for later
    row = 0
    for user, seat in zip(users, seats):
        n = len(seat[0])  # how many new tokens this user has
        if n == 1:
            rows.append(row)
            dec_users.append(user)
            dec_seats.append(seat)
        else:
            out[row : row + n] = attend(q[row : row + n], k[row : row + n], v[row : row + n], pool, user, seat, i)
        row += n
    if rows:
        r = torch.tensor(rows)
        out[r] = attend_decode(q[r], k[r], v[r], pool, dec_users, dec_seats, i)
    return out @ w[p + "o_proj.weight"].T


def attend_decode(q, k, v, pool, users, seats, i):
    # All B users at once, 1 new token each. q is (B, H, D), k and v are (B, KV, D).
    B, H, D = q.shape
    KV = k.shape[1]

    # write everyone's new k and v into their seats, in one go
    pages = torch.cat([seat[0] for seat in seats])
    slots = torch.cat([seat[1] for seat in seats])
    pool["k"][i, pages, slots] = k
    pool["v"][i, pages, slots] = v

    # Read every user's pages. Tables have different lengths, so pad the short
    # ones with page 0, the mask below hides whatever is in there.
    most = max(len(u["table"]) for u in users)
    tables = torch.tensor([u["table"] + [0] * (most - len(u["table"])) for u in users])  # (B, most)
    k = pool["k"][i, tables].reshape(B, -1, KV, D)  # (B, L, KV, D), L = most * 16
    v = pool["v"][i, tables].reshape(B, -1, KV, D)

    group = H // KV
    k = k.repeat_interleave(group, dim=2).transpose(1, 2)  # (B, H, L, D)
    v = v.repeat_interleave(group, dim=2).transpose(1, 2)
    q = q[:, :, None, :]  # (B, H, 1, D)

    scores = q @ k.transpose(-2, -1) / D**0.5  # (B, H, 1, L)

    # No future to hide in decode, only the padding after each user's last token.
    lengths = torch.tensor([u["length"] for u in users])
    padding = torch.arange(k.shape[2])[None, :] >= lengths[:, None]  # (B, L)
    scores = scores.masked_fill(padding[:, None, None, :], float("-inf"))
    out = scores.softmax(dim=-1, dtype=torch.float32).to(v.dtype) @ v  # (B, H, 1, D)
    return out.reshape(B, H * D)


def attend(q, k, v, pool, user, seat, i):
    # One user. q is (T, H, D), k and v are (T, KV, D) for their T new tokens.
    T, H, D = q.shape
    KV = k.shape[1]

    # KV cache: old k and v don't change, so write the new ones into their seats
    # and read back all of this user's pages, in order.
    # We store 8 kv heads here, before the repeat below, so half the memory.
    pages, slots = seat
    pool["k"][i, pages, slots] = k
    pool["v"][i, pages, slots] = v
    S = user["length"]  # all tokens so far
    k = pool["k"][i, user["table"]].reshape(-1, KV, D)[:S]  # last page may be half empty
    v = pool["v"][i, user["table"]].reshape(-1, KV, D)[:S]
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

    return out.transpose(0, 1).reshape(T, H * D)  # heads back side by side


def mlp(x, w, i):
    p = f"model.layers.{i}.mlp."
    gate = F.silu(x @ w[p + "gate_proj.weight"].T)
    up = x @ w[p + "up_proj.weight"].T
    return (gate * up) @ w[p + "down_proj.weight"].T
