from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from qwenserve.config import load_config


# Every weight the model needs, and the shape it should have.
# Shapes are [out, in], the way PyTorch stores a linear layer.
def expected_shapes(c):
    q = c.num_attention_heads * c.head_dim  # all query heads side by side
    kv = c.num_key_value_heads * c.head_dim  # all key (or value) heads
    shapes = {
        "model.embed_tokens.weight": (c.vocab_size, c.hidden_size),
        "model.norm.weight": (c.hidden_size,),
        "lm_head.weight": (c.vocab_size, c.hidden_size),
    }
    for i in range(c.num_hidden_layers):
        p = f"model.layers.{i}."
        shapes[p + "input_layernorm.weight"] = (c.hidden_size,)
        shapes[p + "self_attn.q_proj.weight"] = (q, c.hidden_size)
        shapes[p + "self_attn.k_proj.weight"] = (kv, c.hidden_size)
        shapes[p + "self_attn.v_proj.weight"] = (kv, c.hidden_size)
        shapes[p + "self_attn.o_proj.weight"] = (c.hidden_size, q)
        shapes[p + "self_attn.q_norm.weight"] = (c.head_dim,)
        shapes[p + "self_attn.k_norm.weight"] = (c.head_dim,)
        shapes[p + "post_attention_layernorm.weight"] = (c.hidden_size,)
        shapes[p + "mlp.gate_proj.weight"] = (c.intermediate_size, c.hidden_size)
        shapes[p + "mlp.up_proj.weight"] = (c.intermediate_size, c.hidden_size)
        shapes[p + "mlp.down_proj.weight"] = (c.hidden_size, c.intermediate_size)
    return shapes


# Better to fail here than to get strange text out of the model later.
def check(w, c):
    want = expected_shapes(c)
    missing = want.keys() - w.keys()
    extra = w.keys() - want.keys()  # e.g. a q_proj.bias we don't know how to use
    if missing or extra:
        raise ValueError(f"missing {sorted(missing)[:3]}, extra {sorted(extra)[:3]}")
    for name, shape in want.items():
        if tuple(w[name].shape) != shape:
            raise ValueError(f"{name} is {tuple(w[name].shape)}, expected {shape}")
        # TODO: quantized models (GGUF, FP8, 4-bit) for small edge devices
        if w[name].dtype != torch.bfloat16:
            raise ValueError(f"{name} is {w[name].dtype}, only bf16 is supported")


def load_weights(model_name, c):
    # Only the weight files; we don't need the tokenizer here.
    folder = snapshot_download(model_name, allow_patterns=["*.safetensors"])
    files = sorted(Path(folder).glob("*.safetensors"))
    if not files:
        raise ValueError(f"{model_name} has no .safetensors files (a GGUF repo?)")

    # Big models are split into several files. Names never repeat across
    # files, so we can just pour them all into one dict.
    w = {}
    for f in files:
        w.update(load_file(f))

    # Small Qwen3 files still ship a copy of lm_head, but when the config says
    # tied we use the embedding itself, like Hugging Face does.
    if c.tie_word_embeddings:
        w["lm_head.weight"] = w["model.embed_tokens.weight"]

    check(w, c)
    return w


if __name__ == "__main__":
    name = "Qwen/Qwen3-0.6B"
    c = load_config(name)
    w = load_weights(name, c)
    print(len(w), "tensors")
    print(
        "lm_head is the embedding:",
        w["lm_head.weight"] is w["model.embed_tokens.weight"],
    )
