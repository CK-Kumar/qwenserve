# Qwen3's chat format: turns a list of messages into the one text the model reads.
# Same as the Jinja template in tokenizer_config.json, without tools.


def chat_prompt(messages, thinking=False):
    text = ""
    for m in messages:
        content = m["content"]
        if m["role"] == "assistant":
            # old answers keep only the answer, their thinking is dropped to save tokens
            content = content.split("</think>")[-1].lstrip("\n")
        text += f"<|im_start|>{m['role']}\n{content}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    if not thinking:
        text += "<think>\n\n</think>\n\n"  # thinking already "done" and empty, so it answers right away
    return text
