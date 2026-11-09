import json
from dataclasses import dataclass

from huggingface_hub import hf_hub_download


# Every model on Hugging Face comes with a config.json that says how big it is.
# We only keep the numbers our engine needs. The names match config.json so
# it's easy to check one against the other.
@dataclass
class Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool  # True means lm_head reuses the embedding weights
    eos_token_ids: list  # any of these means the model is done talking
    # sampling the model makers recommend, used when a request doesn't say
    temperature: float
    top_p: float
    top_k: int


def load_config(model_name):
    path = hf_hub_download(model_name, "config.json")
    with open(path) as f:
        raw = json.load(f)

    # For now we only run Qwen3 dense models (0.6B to 32B).
    # TODO: qwen3_moe (Qwen3-30B-A3B), needs expert routing in the MLP
    # TODO: qwen3_5 (Qwen3.5), needs linear attention layers
    # Careful: Qwen3 has attention_bias False, so q, k, v have no bias.
    # If a new model has it True, add "+ bias" to those projections.
    if raw["model_type"] != "qwen3":
        raise ValueError(f"{model_name} is {raw['model_type']}, only qwen3 is supported")

    # The stop tokens live in another file, generation_config.json.
    # Qwen3 gives two: <|im_end|> ends a chat turn, <|endoftext|> ends plain text.
    with open(hf_hub_download(model_name, "generation_config.json")) as f:
        gen = json.load(f)
    eos = gen["eos_token_id"]
    if isinstance(eos, int):  # some models give just one number
        eos = [eos]

    return Config(
        vocab_size=raw["vocab_size"],
        hidden_size=raw["hidden_size"],
        intermediate_size=raw["intermediate_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw["head_dim"],
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=raw["rope_theta"],
        tie_word_embeddings=raw["tie_word_embeddings"],
        eos_token_ids=eos,
        temperature=gen.get("temperature", 1.0),  # if missing: plain sampling, no cut
        top_p=gen.get("top_p", 1.0),
        top_k=gen.get("top_k", 0),
    )


if __name__ == "__main__":
    # Quick check: two models we support and one we don't yet.
    for name in ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B", "Qwen/Qwen3-30B-A3B"]:
        try:
            print(name, load_config(name))
        except ValueError as e:
            print(e)
