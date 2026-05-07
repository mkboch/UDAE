import re
import time
from typing import Any, Dict, Tuple

import torch


STOP_PATTERNS = [
    r"<\|im_end\|>",
    r"<\|eot_id\|>",
    r"<end_of_turn>",
    r"</s>",
    r"\nuser\s*:",
    r"\nassistant\s*:",
    r"\nUser\s*:",
    r"\nAssistant\s*:",
    r"\n//thought.*",
    r"\nThinking Process:.*",
]

THINK_BLOCK_PATTERNS = [
    r"<think>.*?</think>",
    r"<\|startofthink\|>.*?<\|endofthink\|>",
]


def clean_generated_text(text: str) -> str:
    out = text

    for pat in THINK_BLOCK_PATTERNS:
        out = re.sub(pat, "", out, flags=re.DOTALL | re.IGNORECASE)

    for pat in STOP_PATTERNS:
        m = re.search(pat, out, flags=re.DOTALL | re.IGNORECASE)
        if m:
            out = out[:m.start()]

    lines = []
    for line in out.splitlines():
        stripped = line.strip()
        if stripped in {"content", "font"}:
            continue
        lines.append(line)

    out = "\n".join(lines).strip()
    return out


def run_inference(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    do_sample: bool = False,
) -> Tuple[str, float, int]:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)

    extra_stop_tokens = ["<|im_end|>", "<|eot_id|>", "<end_of_turn>"]
    for tok in extra_stop_tokens:
        try:
            tok_ids = tokenizer.encode(tok, add_special_tokens=False)
            if len(tok_ids) == 1:
                eos_ids.append(tok_ids[0])
        except Exception:
            pass

    eos_ids = sorted(set(eos_ids)) if eos_ids else None

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }

    if eos_ids:
        gen_kwargs["eos_token_id"] = eos_ids

    if do_sample and temperature is not None:
        gen_kwargs["temperature"] = temperature

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    torch.cuda.synchronize()
    latency_sec = time.perf_counter() - t0

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = output[0][prompt_len:]
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    generated_text = clean_generated_text(generated_text)
    n_new_tokens = int(gen_ids.shape[0])

    return generated_text, latency_sec, n_new_tokens


def tokens_per_second(n_tokens: int, latency_sec: float) -> float:
    if latency_sec <= 0:
        return 0.0
    return float(n_tokens) / float(latency_sec)
