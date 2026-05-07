from __future__ import annotations

import gc
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_torch_dtype(dtype_name: str):
    name = str(dtype_name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.bfloat16


def unload_model(model) -> None:
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_model_kwargs(model_cfg: Dict[str, Any], mode: str) -> Dict[str, Any]:
    revision = model_cfg.get("revision", None)
    dtype_name = model_cfg.get("default_dtype", "bfloat16")
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    torch_dtype = get_torch_dtype(dtype_name)

    kwargs = dict(
        revision=revision,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )

    if mode == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["dtype"] = torch_dtype

    return kwargs


def load_model(model_cfg: Dict[str, Any]) -> Tuple[Any, Any, float, str]:
    model_id = model_cfg["hf_id"]
    revision = model_cfg.get("revision", None)
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    attempted = []
    last_err = None

    for mode in ["bf16", "4bit"]:
        attempted.append(mode)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **_build_model_kwargs(model_cfg, mode),
            )
            model.eval()
            if hasattr(model, "generation_config") and model.generation_config is not None:
                model.generation_config.do_sample = False
                model.generation_config.temperature = None
                model.generation_config.top_p = None
                model.generation_config.top_k = None

            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            return model, tokenizer, peak_vram_gb, mode
        except Exception as e:
            last_err = e
            unload_model(locals().get("model", None))

    raise RuntimeError(f"Could not load {model_id}. Attempts={attempted}. Last error={last_err}")
