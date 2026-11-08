import torch

# Chooses the next word.
# Low temperature plays it safe, high temperature takes risks.
# top_k and top_p throw away the silly choices first.


def sample(logits, temperature, generator=None, top_k=0, top_p=1.0):
    # logits: (vocab,) scores for one user's next token
    # top_k 0 and top_p 1.0 mean no cut
    if temperature == 0:
        return int(logits.argmax())  # greedy
    probs = (logits.float() / temperature).softmax(dim=-1)

    # best first, so cutting the bad tail is just cutting the end of the list
    probs, ids = probs.sort(descending=True)
    if top_k > 0:
        probs, ids = probs[:top_k], ids[:top_k]
        probs = probs / probs.sum()  # make them add up to 1 again
    if top_p < 1.0:
        # keep tokens until they add up to top_p, the one that crosses the line stays
        before = probs.cumsum(dim=0) - probs
        keep = before < top_p
        probs, ids = probs[keep], ids[keep]

    # pick one token at random, weighted by probs
    pick = torch.multinomial(probs, 1, generator=generator)
    return int(ids[pick])
