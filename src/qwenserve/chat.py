# Qwen3's chat format: turns a list of messages into the one text the model reads.
# Same as the Jinja template in tokenizer_config.json, without tools.
# And the way back: answer tokens into text, piece by piece for streaming.


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


def new_text(tok, ids, sent, done=False):
    # For streaming: ids are all answer tokens so far, sent is the text already
    # sent. Returns the text to send now.
    # One character can be split over tokens, 🚀 is 3 tokens and each one alone
    # decodes to "�". So decode everything so far, and if it ends in "�", wait
    # for the next token. At the end (done) send what is left anyway.
    text = tok.decode(ids, skip_special_tokens=True)
    if text.endswith("�") and not done:
        return ""
    return text[len(sent) :]
