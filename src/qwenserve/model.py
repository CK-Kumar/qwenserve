import time

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


def generate_batch(prompts, w, c, max_new_tokens, temperature=0, top_k=0, top_p=1.0, seed=None, max_running=None, num_pages=None, log=False, stats_every=10):
    # prompts is a list of token id lists, one per user. Runs until everyone is done.
    # A user stops at eos or after max_new_tokens, the others go on.
    if max_running is None:
        max_running = len(prompts)  # everyone gets a seat
    if num_pages is None:
        num_pages = sum(pages_needed(ids, max_new_tokens) for ids in prompts)  # everyone fits
    engine = make_engine(w, c, num_pages, max_running, log, stats_every)
    requests = [add_request(engine, ids, max_new_tokens, temperature, top_k, top_p, seed) for ids in prompts]
    while engine["running"] or engine["waiting"]:
        step(engine)
    return [r["text"] for r in requests]


def pages_needed(ids, max_new_tokens):
    # pages if the answer runs all the way to max_new_tokens
    return (len(ids) + max_new_tokens) // PAGE + 1


def make_engine(w, c, num_pages, max_running, log=False, stats_every=10):
    # The kitchen, made once: one pool and two lines. Users come and go, it stays.
    # At most max_running users talk at once, the rest wait for a free seat.
    # A user also waits until their worst case fits in the pool.
    return {
        "w": w,
        "c": c,
        "pool": make_pool(c, num_pages, w["model.embed_tokens.weight"].dtype),
        "num_pages": num_pages,
        "max_running": max_running,
        # Pages promised to running users. They get them bit by bit as the answer
        # grows, so we can't just count pool["free"].
        "reserved": 0,
        "waiting": [],  # first come, first served
        "running": [],
        "next_id": 0,  # only for the log
        "log": log,
        "step": 0,
        # for the stats line: tokens done since the last one
        "stats_every": stats_every,
        "last": time.time(),
        "prompt_toks": 0,
        "gen_toks": 0,
    }


def add_request(engine, ids, max_new_tokens, temperature=0, top_k=0, top_p=1.0, seed=None):
    # Put one request in the waiting line. One dict per request keeps its stuff
    # together, so users can join and leave without lining up lists by number.
    r = {
        "id": engine["next_id"],  # only for the log
        "prompt": ids,
        "text": list(ids),  # full text so far, to return at the end
        "new": list(ids),  # first step sends the whole prompt (prefill)
        "user": new_user(),  # their page table + length
        "max_new_tokens": max_new_tokens,
        "need": pages_needed(ids, max_new_tokens),
        # Sampling, each request its own. temperature 0 is greedy. Its own seeded
        # generator, so the same seed gives the same text whoever else is running.
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "generator": torch.Generator().manual_seed(seed) if seed is not None else None,
    }
    if r["need"] > engine["num_pages"]:
        raise ValueError("this request needs more pages than the whole pool, it would wait forever")
    engine["next_id"] += 1
    engine["waiting"].append(r)
    return r


def step(engine):
    # One round: fill free seats, 1 new token for every running user, finished
    # users out. Returns the requests that finished in this round.
    w, c, pool, waiting, log = engine["w"], engine["c"], engine["pool"], engine["waiting"], engine["log"]
    running = engine["running"]
    engine["step"] += 1
    s = engine["step"]

    # fill free seats before the step, so a new user joins while others are mid-answer
    while engine["max_running"] > len(running) and waiting and engine["reserved"] + waiting[0]["need"] <= engine["num_pages"]:
        r = waiting.pop(0)
        running.append(r)
        engine["reserved"] += r["need"]
        engine["prompt_toks"] += len(r["prompt"])
        if log:
            print(f"step {s}: user {r['id']} joins ({len(r['prompt'])} prompt tokens)")
    if not running:
        return []  # nothing to cook

    counts = [len(r["new"]) for r in running]
    flat = [t for r in running for t in r["new"]]  # running users' new tokens in one list
    with torch.no_grad():
        logits = forward(torch.tensor(flat), w, c, pool, [r["user"] for r in running], counts)

    # each request's next token comes from its last row
    still, finished = [], []
    row = 0
    for r, n in zip(running, counts):
        row += n
        next_id = sample(logits[row - 1], r["temperature"], r["generator"], r["top_k"], r["top_p"])
        r["text"].append(next_id)
        r["new"] = [next_id]  # after prefill, 1 token per user
        if next_id in c.eos_token_ids or len(r["text"]) - len(r["prompt"]) == r["max_new_tokens"]:
            free(pool, r["user"])
            engine["reserved"] -= r["need"]
            finished.append(r)
            if log:
                reason = "eos" if next_id in c.eos_token_ids else "length"
                print(f"step {s}: user {r['id']} done, {reason} ({len(r['text']) - len(r['prompt'])} new tokens)")
        else:
            still.append(r)
    engine["gen_toks"] += len(counts)  # each running user got 1 new token
    engine["running"] = still

    # one summary line every stats_every seconds, like vLLM's
    secs = time.time() - engine["last"]
    if log and secs >= engine["stats_every"]:
        used = engine["num_pages"] - len(pool["free"])
        print(
            f"prompt {engine['prompt_toks'] / secs:.1f} tok/s, generation {engine['gen_toks'] / secs:.1f} tok/s, "
            f"running {len(still)}, waiting {len(waiting)}, KV pool {100 * used / engine['num_pages']:.1f}%"
        )
        engine["last"] = time.time()
        engine["prompt_toks"] = engine["gen_toks"] = 0
    return finished
