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


def generate_batch(prompts, w, c, max_new_tokens, temperature=0, top_k=0, top_p=1.0, seed=None, max_running=None, num_pages=None, log=False):
    # prompts is a list of token id lists, one per user.
    # temperature 0 is greedy. Same seed gives the same text.
    # A user stops at eos or after max_new_tokens, the others go on.
    # At most max_running users talk at once, the rest wait for a free seat.
    # The pool has num_pages pages. A user also waits until their worst case fits.
    # One dict per request keeps its stuff together, so users can join and leave
    # without lining up separate lists by number.
    requests = [
        {
            "id": i,  # only for the log
            "prompt": ids,
            "text": list(ids),  # full text so far, to return at the end
            "new": list(ids),  # first step sends the whole prompt (prefill)
            "user": new_user(),  # their page table + length
            "need": (len(ids) + max_new_tokens) // PAGE + 1,  # pages if the answer runs to max_new_tokens
        }
        for i, ids in enumerate(prompts)
    ]
    if max_running is None:
        max_running = len(requests)  # everyone gets a seat
    if num_pages is None:
        num_pages = sum(r["need"] for r in requests)  # everyone fits
    if max(r["need"] for r in requests) > num_pages:
        raise ValueError("one user needs more pages than the whole pool, they would wait forever")
    pool = make_pool(c, num_pages, w["model.embed_tokens.weight"].dtype)
    # Pages promised to running users. They get them bit by bit as the answer
    # grows, so we can't just count pool["free"].
    reserved = 0
    waiting = list(requests)  # first come, first served
    running = []
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    step = 0
    with torch.no_grad():
        while running or waiting:
            step += 1
            # fill free seats before the step, so a new user joins while others are mid-answer
            while max_running > len(running) and waiting and reserved + waiting[0]["need"] <= num_pages:
                r = waiting.pop(0)
                running.append(r)
                reserved += r["need"]
                if log:
                    print(f"step {step}: user {r['id']} joins ({len(r['prompt'])} prompt tokens)")
            if log:
                used = num_pages - len(pool["free"])
                print(f"step {step}: running {len(running)}, waiting {len(waiting)}, pages used {used}, reserved {reserved}/{num_pages}")

            counts = [len(r["new"]) for r in running]
            flat = [t for r in running for t in r["new"]]  # running users' new tokens in one list
            logits = forward(torch.tensor(flat), w, c, pool, [r["user"] for r in running], counts)

            # each request's next token comes from its last row
            still = []
            row = 0
            for r, n in zip(running, counts):
                row += n
                next_id = sample(logits[row - 1], temperature, generator, top_k, top_p)
                r["text"].append(next_id)
                r["new"] = [next_id]  # after prefill, 1 token per user
                if next_id in c.eos_token_ids or len(r["text"]) - len(r["prompt"]) == max_new_tokens:
                    free(pool, r["user"])
                    reserved -= r["need"]
                    if log:
                        reason = "eos" if next_id in c.eos_token_ids else "length"
                        print(f"step {step}: user {r['id']} done, {reason} ({len(r['text']) - len(r['prompt'])} new tokens)")
                else:
                    still.append(r)
            running = still
    return [r["text"] for r in requests]
