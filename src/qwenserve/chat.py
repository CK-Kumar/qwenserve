# Qwen3's chat format: turns a list of messages into the one text the model reads.
# Same as the Jinja template in tokenizer_config.json, without tools.
# Also new_text: tokens back to text, for streaming.


def chat_prompt(messages, thinking=False):
    text = ""
    for m in messages:
        content = m["content"]
        if m["role"] == "assistant":
            # drop the thinking from old answers, like Qwen's template
            content = content.split("</think>")[-1].lstrip("\n")
        text += f"<|im_start|>{m['role']}\n{content}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    if not thinking:
        text += "<think>\n\n</think>\n\n"  # empty think block = thinking off
    return text


def new_text(tok, ids, sent, done=False):
    # Text to send now, from all answer tokens so far and the text already sent.
    # A character can span tokens (🚀 is 3) and alone they decode to "�",
    # so wait until it's complete. When done, send whatever is left.
    text = tok.decode(ids, skip_special_tokens=True)
    if text.endswith("�") and not done:
        return ""
    return text[len(sent) :]
