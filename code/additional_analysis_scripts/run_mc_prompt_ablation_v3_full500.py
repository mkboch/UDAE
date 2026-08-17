from pathlib import Path
from typing import Any, Dict, List, Tuple
import csv, json, math, os, re, sys, time, traceback
import pandas as pd
import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from models.loader import load_model as _project_load_model, unload_model as _project_unload_model
from models.inference import run_inference

# ---------------------------------------------------------------------
# Stronger tokenizer compatibility patch for current Gemma tokenizer
# files under the local transformers build.
#
# The fast tokenizer currently fails because extra_special_tokens can be
# a list, while transformers expects a dict-like object with .keys().
# The slow tokenizer fallback can also fail with SentencePiece
# "TypeError: not a string". Therefore we keep the fast tokenizer path
# and normalize the special-token structure before initialization.
# ---------------------------------------------------------------------
try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    _ORIG_SET_MODEL_SPECIFIC_SPECIAL_TOKENS = PreTrainedTokenizerBase._set_model_specific_special_tokens

    def _safe_set_model_specific_special_tokens(self, special_tokens):
        if isinstance(special_tokens, list):
            # These are model-specific extra aliases. For our benchmark,
            # we do not need to expose them as tokenizer attributes, and
            # treating them as an empty dict avoids the incompatible
            # list.keys() call while preserving ordinary special tokens.
            special_tokens = {}
        return _ORIG_SET_MODEL_SPECIFIC_SPECIAL_TOKENS(self, special_tokens)

    PreTrainedTokenizerBase._set_model_specific_special_tokens = _safe_set_model_specific_special_tokens
    print("PATCH ACTIVE: normalized list-valued extra_special_tokens for tokenizer loading", flush=True)
except Exception as _e:
    print(f"TOKENIZER SPECIAL TOKEN PATCH WARNING: {_e}", flush=True)

try:
    from transformers import AutoTokenizer
    _ORIG_AUTO_TOKENIZER_FROM_PRETRAINED = AutoTokenizer.from_pretrained

    def _safe_auto_tokenizer_from_pretrained(*args, **kwargs):
        # Prefer fast tokenizer. The special-token patch above fixes the
        # known Gemma fast-tokenizer list.keys() error. Do not force
        # use_fast=False because the slow Gemma tokenizer can fail with
        # SentencePiece TypeError when vocab_file is not a plain string.
        try:
            kwargs2 = dict(kwargs)
            if "gemma" in str(args[0] if args else kwargs.get("pretrained_model_name_or_path", "")).lower():
                kwargs2.pop("use_fast", None)
                kwargs2["use_fast"] = True
            return _ORIG_AUTO_TOKENIZER_FROM_PRETRAINED(*args, **kwargs2)
        except AttributeError as e:
            msg = str(e)
            model_id = str(args[0]) if args else str(kwargs.get("pretrained_model_name_or_path", ""))
            print(f"TOKENIZER_LOAD_ATTRIBUTE_ERROR for {model_id}: {type(e).__name__}: {e}", flush=True)
            raise

    AutoTokenizer.from_pretrained = _safe_auto_tokenizer_from_pretrained
    print("PATCH ACTIVE: AutoTokenizer keeps Gemma fast-tokenizer path", flush=True)
except Exception as _tok_patch_e:
    print(f"TOKENIZER AUTO PATCH WARNING: {_tok_patch_e}", flush=True)



from prompts.builder import maybe_apply_chat_template, get_generation_config


# ---------------------------------------------------------------------
# Clean V2 loader: bf16-only, no 4-bit fallback.
# This avoids ambiguity about quantization in the prompt-ablation evidence.
# Run this script from an environment whose transformers version supports
# all target model architectures, especially Gemma-4.
# ---------------------------------------------------------------------
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_cfg):
    model_id = model_cfg["hf_id"]
    revision = model_cfg.get("revision", None)
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f"BF16_DIRECT_LOAD tokenizer={model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"BF16_DIRECT_LOAD model={model_id}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)

    return model, tokenizer, peak_vram_gb, "bf16_direct_no_4bit"

def unload_model(model):
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

OUT = ROOT / "review_response_runs" / "mc_prompt_ablation_v3_full500"
RAW = ROOT / "results" / "raw"
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)

TAG = os.environ.get("TAG", "mc_prompt_ablation_v3_full500")
ORIG_TAG = os.environ.get("ORIG_TAG", "journal_full500_v1")
LIMIT = int(os.environ.get("LIMIT", "500"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))

MODELS = os.environ.get(
    "MODELS",
    "gemma_4_26b_a4b,gemma_4_e4b,qwen3_8b,phi_4_reasoning"
).split(",")

DATASETS = ["arc_challenge", "truthfulqa_mc1"]

BASELINE_MAP = {
    "orig_zero_shot_answer_only": "zero_shot",
    "orig_cot_answer_only": "cot",
    "orig_few_shot_cot_answer_only": "few_shot_cot",
}

NEW_STRATEGIES = [
    "cot_rationale_final_letter",
    "few_shot_cot_rationale_final_letter",
]

ALL_ABLATION_STRATEGIES = list(BASELINE_MAP.keys()) + NEW_STRATEGIES

RUN_COLUMNS = [
    "timestamp_utc",
    "requested_model_name",
    "actual_model_name",
    "model_pretty_name",
    "hf_id",
    "architecture",
    "total_params_b",
    "active_params_b",
    "load_mode",
    "dataset_name",
    "strategy",
    "sample_id",
    "question",
    "gold_answer",
    "prediction",
    "diagnostic_lenient_prediction",
    "correct",
    "diagnostic_lenient_correct",
    "latency_sec",
    "n_output_tokens",
    "tokens_per_sec",
    "peak_vram_gb",
    "response_text",
    "prompt_text",
    "error",
]

def log(x: str = ""):
    print(x, flush=True)

def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def load_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows

def load_records_for_ablation(model_name: str, dataset_name: str, limit: int) -> List[Dict[str, Any]]:
    """
    V3 full-500 ablation must use the same item set as journal_full500_v1.
    The older data/prepared/*.jsonl files may contain only the submitted matched
    238 examples, so for ARC/TruthfulQA full-500 prompt ablation we load
    sample_id/question/gold_answer directly from the journal_full500_v1
    zero-shot baseline CSV for the same model and dataset.
    """
    baseline = RAW / f"{ORIG_TAG}__{model_name}__{dataset_name}__zero_shot__n{limit}.csv"
    if baseline.exists():
        df = pd.read_csv(baseline).head(limit)
        need = {"sample_id", "question", "gold_answer"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"Baseline file missing required columns {missing}: {baseline}")
        return [
            {
                "sample_id": str(r["sample_id"]),
                "question": str(r["question"]),
                "gold_answer": str(r["gold_answer"]),
            }
            for _, r in df.iterrows()
        ]

    # Fallback only for non-full settings.
    rows = load_jsonl(ROOT / "data" / "prepared" / f"{dataset_name}.jsonl", limit)
    if len(rows) < limit:
        raise ValueError(
            f"Could not load {limit} records for {model_name}/{dataset_name}. "
            f"Missing baseline {baseline}; fallback prepared JSONL has only {len(rows)} rows."
        )
    return rows[:limit]

def load_model_configs() -> Dict[str, Dict[str, Any]]:
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    return {m["name"]: m for m in cfg["models"]}

def load_few_shot_examples() -> Dict[str, List[Dict[str, str]]]:
    p = ROOT / "prompts" / "few_shot_examples.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())

def normalize_mc(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    m = re.search(r"\b([A-E])\b", str(x).strip().upper())
    return m.group(1) if m else ""

def strip_traces(text: Any) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    s = str(text)
    s = s.replace("[object Object]", " ")
    s = re.sub(r"<think>.*?</think>", " ", s, flags=re.I | re.S)
    s = re.sub(r"<\s*/?\s*think\s*>", " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_final_hash_letter(text: Any) -> str:
    s = "" if text is None or (isinstance(text, float) and math.isnan(text)) else str(text)
    hits = re.findall(r"####\s*([A-E])\b", s, flags=re.I)
    return hits[-1].upper() if hits else ""

def extract_lenient_letter(text: Any) -> str:
    s = strip_traces(text)
    if not s:
        return ""
    patterns = [
        r"####\s*([A-E])\b",
        r"(?:final\s+answer|answer|option|choice)\s*(?:is|:)?\s*([A-E])\b",
        r"\b([A-E])\b",
    ]
    for pat in patterns:
        hits = re.findall(pat, s, flags=re.I)
        if hits:
            return hits[-1].upper()
    return ""

def is_correct(pred: str, gold: Any) -> int:
    g = normalize_mc(gold)
    return int(pred != "" and g != "" and pred == g)

def tps(n_tokens: Any, latency: Any) -> float:
    try:
        n = float(n_tokens)
        l = float(latency)
        return n / l if l > 0 else 0.0
    except Exception:
        return 0.0

def output_path(model_name: str, dataset_name: str, strategy: str, n: int) -> Path:
    return RAW / f"{TAG}__{model_name}__{dataset_name}__{strategy}__n{n}.csv"

def original_path(model_name: str, dataset_name: str, orig_strategy: str, n: int) -> Path:
    return RAW / f"{ORIG_TAG}__{model_name}__{dataset_name}__{orig_strategy}__n{n}.csv"

def safe_response_series(df: pd.DataFrame) -> pd.Series:
    if "response_text" not in df.columns:
        return pd.Series([""] * len(df))
    return df["response_text"].fillna("").astype(str)

def build_few_shot_block(dataset_name: str) -> str:
    examples = load_few_shot_examples().get(dataset_name, [])
    blocks = []
    for ex in examples:
        q = ex.get("question", "").strip()
        reasoning = ex.get("reasoning", "").strip()
        ans = normalize_mc(ex.get("answer", ""))
        if not q or not ans:
            continue
        if reasoning:
            blocks.append(f"Q: {q}\nA: {reasoning}\n#### {ans}")
        else:
            blocks.append(f"Q: {q}\nA:\n#### {ans}")
    return "\n\n".join(blocks)

def build_mc_prompt(strategy: str, question: str, dataset_name: str) -> str:
    if strategy == "cot_rationale_final_letter":
        return (
            f"{question}\n\n"
            "Let's reason briefly before answering.\n\n"
            "Rules:\n"
            "1. You may write a short rationale in no more than 4 short steps.\n"
            "2. The final line must be exactly: #### <letter>\n"
            "3. Valid final answers are A, B, C, D, or E.\n"
            "4. Do not output anything after the final line."
        )

    if strategy == "few_shot_cot_rationale_final_letter":
        fs = build_few_shot_block(dataset_name)
        prefix = f"{fs}\n\n" if fs.strip() else ""
        return (
            f"{prefix}"
            f"Q: {question}\n"
            "A: Let's reason briefly before answering.\n\n"
            "Rules:\n"
            "1. You may write a short rationale in no more than 4 short steps.\n"
            "2. The final line must be exactly: #### <letter>\n"
            "3. Valid final answers are A, B, C, D, or E.\n"
            "4. Do not output anything after the final line."
        )

    raise ValueError(f"Unknown ablation strategy: {strategy}")

def copy_baseline_condition(model_name: str, dataset_name: str, ablation_strategy: str, orig_strategy: str):
    src = original_path(model_name, dataset_name, orig_strategy, LIMIT)
    dst = output_path(model_name, dataset_name, ablation_strategy, LIMIT)

    if dst.exists():
        try:
            if len(pd.read_csv(dst)) >= LIMIT:
                log(f"SKIP baseline exists: {dst}")
                return
        except Exception:
            pass

    if not src.exists():
        raise FileNotFoundError(f"Missing original baseline file: {src}")

    df = pd.read_csv(src)
    df = df.head(LIMIT).copy()
    df["strategy"] = ablation_strategy

    # Add diagnostic columns for consistent summary.
    resp = safe_response_series(df)
    df["diagnostic_lenient_prediction"] = resp.apply(extract_lenient_letter)
    df["diagnostic_lenient_correct"] = [
        is_correct(p, g) for p, g in zip(df["diagnostic_lenient_prediction"], df["gold_answer"])
    ]
    if "prompt_text" not in df.columns:
        df["prompt_text"] = ""
    if "error" not in df.columns:
        df["error"] = ""

    for c in RUN_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    df[RUN_COLUMNS].to_csv(dst, index=False)
    log(f"WROTE baseline copy: {dst}")

def existing_sample_ids(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["sample_id"])
        return set(df["sample_id"].astype(str))
    except Exception:
        return set()

def append_row(path: Path, row: Dict[str, Any]):
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8", buffering=1) as f:
        w = csv.DictWriter(f, fieldnames=RUN_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in RUN_COLUMNS})

def condition_complete(path: Path, n: int) -> bool:
    if not path.exists():
        return False
    try:
        return len(pd.read_csv(path)) >= n
    except Exception:
        return False

def run_new_condition(model_name: str, model_cfg: Dict[str, Any], model: Any, tokenizer: Any,
                      peak_vram_gb: float, load_mode: str, dataset_name: str, strategy: str):
    records = load_records_for_ablation(model_name, dataset_name, LIMIT)
    out_path = output_path(model_name, dataset_name, strategy, len(records))

    if condition_complete(out_path, len(records)):
        log(f"SKIP complete: {out_path}")
        return

    done = existing_sample_ids(out_path)
    log(f"\n--- RUN NEW model={model_name} dataset={dataset_name} strategy={strategy} n={len(records)} already_done={len(done)} ---")

    gen_cfg = get_generation_config()
    temperature = float(gen_cfg.get("temperature", 0.0))
    do_sample = bool(gen_cfg.get("do_sample", False))

    for i, rec in enumerate(records, start=1):
        sid = str(rec.get("sample_id", i))
        if sid in done:
            continue

        question = rec["question"]
        gold = rec["gold_answer"]
        prompt = build_mc_prompt(strategy, question, dataset_name)
        rendered_prompt = maybe_apply_chat_template(
            tokenizer=tokenizer,
            prompt=prompt,
            use_chat_template=bool(model_cfg.get("use_chat_template", True)),
        )

        response_text = ""
        latency_sec = 0.0
        n_output_tokens = 0
        pred = ""
        pred_lenient = ""
        correct = 0
        correct_lenient = 0
        err = ""

        try:
            response_text, latency_sec, n_output_tokens = run_inference(
                model=model,
                tokenizer=tokenizer,
                prompt=rendered_prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=temperature,
                do_sample=do_sample,
            )
            pred = extract_final_hash_letter(response_text)
            pred_lenient = extract_lenient_letter(response_text)
            correct = is_correct(pred, gold)
            correct_lenient = is_correct(pred_lenient, gold)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            response_text = traceback.format_exc()

        row = {
            "timestamp_utc": utc_now(),
            "requested_model_name": model_name,
            "actual_model_name": model_cfg.get("name", model_name),
            "model_pretty_name": model_cfg.get("pretty_name", model_name),
            "hf_id": model_cfg.get("hf_id", ""),
            "architecture": model_cfg.get("architecture", ""),
            "total_params_b": model_cfg.get("total_params_b", ""),
            "active_params_b": model_cfg.get("active_params_b", ""),
            "load_mode": load_mode,
            "dataset_name": dataset_name,
            "strategy": strategy,
            "sample_id": sid,
            "question": question,
            "gold_answer": gold,
            "prediction": pred,
            "diagnostic_lenient_prediction": pred_lenient,
            "correct": correct,
            "diagnostic_lenient_correct": correct_lenient,
            "latency_sec": latency_sec,
            "n_output_tokens": n_output_tokens,
            "tokens_per_sec": tps(n_output_tokens, latency_sec),
            "peak_vram_gb": peak_vram_gb,
            "response_text": response_text,
            "prompt_text": prompt,
            "error": err,
        }
        append_row(out_path, row)

        log(
            f"[{i}/{len(records)}] model={model_name} dataset={dataset_name} strategy={strategy} "
            f"sample_id={sid} pred={pred or '<missing>'} gold={gold} correct={correct} "
            f"lenient={pred_lenient or '<missing>'}/{correct_lenient} latency={latency_sec:.3f}s error={'YES' if err else 'NO'}"
        )

def summarize_outputs():
    rows = []
    for model_name in MODELS:
        for dataset_name in DATASETS:
            for strategy in ALL_ABLATION_STRATEGIES:
                p = output_path(model_name, dataset_name, strategy, LIMIT)
                if not p.exists():
                    rows.append({
                        "model": model_name, "dataset": dataset_name, "strategy": strategy,
                        "present": 0, "n": 0
                    })
                    continue
                df = pd.read_csv(p)
                resp = safe_response_series(df)
                pred = df.get("prediction", pd.Series([""] * len(df))).fillna("").astype(str)
                correct = pd.to_numeric(df.get("correct", pd.Series([0] * len(df))), errors="coerce").fillna(0).astype(int)
                lenient_correct = pd.to_numeric(df.get("diagnostic_lenient_correct", pd.Series([0] * len(df))), errors="coerce").fillna(0).astype(int)
                err = df.get("error", pd.Series([""] * len(df))).fillna("").astype(str)
                final_hash = resp.str.contains(r"####\s*[A-E]\b", regex=True, flags=re.I)
                bare_letter = resp.apply(lambda x: bool(re.match(r"^\s*[A-E]\s*$", str(x), flags=re.I)))
                think_tag = resp.apply(lambda x: bool(re.search(r"<\s*/?\s*think\s*>", str(x), flags=re.I)))
                rows.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "strategy": strategy,
                    "present": 1,
                    "n": len(df),
                    "strict_acc": float(correct.mean()) if len(df) else np.nan,
                    "diagnostic_lenient_acc": float(lenient_correct.mean()) if len(df) else np.nan,
                    "missing_prediction_rate": float(pred.apply(lambda x: str(x).strip() == "").mean()) if len(df) else np.nan,
                    "final_hash_answer_rate": float(final_hash.mean()) if len(df) else np.nan,
                    "bare_letter_response_rate": float(bare_letter.mean()) if len(df) else np.nan,
                    "think_tag_rate": float(think_tag.mean()) if len(df) else np.nan,
                    "median_response_chars": float(resp.str.len().median()) if len(df) else np.nan,
                    "mean_latency_sec": float(pd.to_numeric(df.get("latency_sec", pd.Series(dtype=float)), errors="coerce").mean()) if len(df) else np.nan,
                    "error_rows": int((err.str.strip() != "").sum()),
                    "file": str(p),
                })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "mc_prompt_ablation_condition_summary.csv", index=False)
    (OUT / "mc_prompt_ablation_condition_summary.md").write_text(summary.to_markdown(index=False))

    pairs = []
    for model_name in MODELS:
        for dataset_name in DATASETS:
            sub = summary[(summary["model"] == model_name) & (summary["dataset"] == dataset_name)]
            def row(strategy):
                r = sub[sub["strategy"] == strategy]
                return None if r.empty else r.iloc[0]

            for old_s, new_s in [
                ("orig_cot_answer_only", "cot_rationale_final_letter"),
                ("orig_few_shot_cot_answer_only", "few_shot_cot_rationale_final_letter"),
            ]:
                a = row(old_s)
                b = row(new_s)
                if a is None or b is None:
                    continue
                pairs.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "comparison": f"{old_s} -> {new_s}",
                    "old_strict_acc": a["strict_acc"],
                    "new_strict_finalline_acc": b["strict_acc"],
                    "delta_new_minus_old_strict": b["strict_acc"] - a["strict_acc"],
                    "new_diagnostic_lenient_acc": b["diagnostic_lenient_acc"],
                    "old_missing_prediction_rate": a["missing_prediction_rate"],
                    "new_missing_prediction_rate": b["missing_prediction_rate"],
                    "new_final_hash_answer_rate": b["final_hash_answer_rate"],
                    "old_median_response_chars": a["median_response_chars"],
                    "new_median_response_chars": b["median_response_chars"],
                    "old_think_tag_rate": a["think_tag_rate"],
                    "new_think_tag_rate": b["think_tag_rate"],
                    "new_error_rows": b["error_rows"],
                })

    pair_df = pd.DataFrame(pairs)
    pair_df.to_csv(OUT / "mc_prompt_ablation_pairwise_comparison.csv", index=False)
    (OUT / "mc_prompt_ablation_pairwise_comparison.md").write_text(pair_df.to_markdown(index=False))

    agg = summary.groupby(["dataset", "strategy"], dropna=False).agg(
        mean_strict_acc=("strict_acc", "mean"),
        mean_diagnostic_lenient_acc=("diagnostic_lenient_acc", "mean"),
        mean_missing_prediction_rate=("missing_prediction_rate", "mean"),
        mean_final_hash_answer_rate=("final_hash_answer_rate", "mean"),
        mean_bare_letter_response_rate=("bare_letter_response_rate", "mean"),
        mean_think_tag_rate=("think_tag_rate", "mean"),
        mean_median_response_chars=("median_response_chars", "mean"),
        total_error_rows=("error_rows", "sum"),
    ).reset_index()
    agg.to_csv(OUT / "mc_prompt_ablation_grouped_summary.csv", index=False)
    (OUT / "mc_prompt_ablation_grouped_summary.md").write_text(agg.to_markdown(index=False))

    status_lines = []
    status_lines.append("# MC Prompt Ablation V3 Full-500 Status")
    status_lines.append("")
    status_lines.append(f"- tag: `{TAG}`")
    status_lines.append(f"- original tag reused: `{ORIG_TAG}`")
    status_lines.append(f"- limit: `{LIMIT}`")
    status_lines.append(f"- max_new_tokens for rationale prompts: `{MAX_NEW_TOKENS}`")
    status_lines.append(f"- models: `{', '.join(MODELS)}`")
    status_lines.append(f"- datasets: `{', '.join(DATASETS)}`")
    status_lines.append("")
    status_lines.append("## Grouped summary")
    status_lines.append("")
    status_lines.append(agg.to_markdown(index=False))
    status_lines.append("")
    status_lines.append("## Pairwise comparison")
    status_lines.append("")
    status_lines.append(pair_df.to_markdown(index=False))
    status_lines.append("")
    status_lines.append("## Interpretation for revision")
    status_lines.append("")
    status_lines.append(
        "Use this ablation to answer the MC prompt-protocol concern directly: the original MC CoT conditions were answer-format-constrained, "
        "and the revised rationale-allowed conditions test whether allowing a short rationale followed by a strict final `#### <letter>` line changes "
        "accuracy, scoreability, final-answer compliance, and output length."
    )
    (OUT / "MC_PROMPT_ABLATION_V3_FULL500_STATUS.md").write_text("\n".join(status_lines))

    return summary, pair_df, agg

def main():
    log("============================================================")
    log("MC PROMPT ABLATION V3 FULL500")
    log("============================================================")
    log(f"ROOT={ROOT}")
    log(f"TAG={TAG}")
    log(f"ORIG_TAG={ORIG_TAG}")
    log(f"LIMIT={LIMIT}")
    log(f"MAX_NEW_TOKENS={MAX_NEW_TOKENS}")
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    log("")

    try:
        smi = os.popen("nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits").read()
        log("GPU snapshot:")
        log(smi.strip())
    except Exception:
        pass

    model_cfgs = load_model_configs()

    log("\n[1] Copying original answer-only MC baselines into ablation tag...")
    for model_name in MODELS:
        for dataset_name in DATASETS:
            for ablation_strategy, orig_strategy in BASELINE_MAP.items():
                copy_baseline_condition(model_name, dataset_name, ablation_strategy, orig_strategy)

    log("\n[2] Running rationale-allowed MC prompt conditions...")
    any_errors = False
    for model_name in MODELS:
        if model_name not in model_cfgs:
            raise ValueError(f"Unknown model in configs/models.yaml: {model_name}")
        cfg = model_cfgs[model_name]

        pending = []
        for dataset_name in DATASETS:
            for strategy in NEW_STRATEGIES:
                records = load_records_for_ablation(model_name, dataset_name, LIMIT)
                p = output_path(model_name, dataset_name, strategy, len(records))
                if not condition_complete(p, len(records)):
                    pending.append((dataset_name, strategy))

        if not pending:
            log(f"SKIP loading {model_name}: all new ablation conditions complete.")
            continue

        log(f"\n=== Loading model {model_name} for {len(pending)} pending conditions ===")
        model = tokenizer = None
        peak_vram_gb = ""
        load_mode = ""
        try:
            model, tokenizer, peak_vram_gb, load_mode = load_model(cfg)
            log(f"Loaded {model_name}: load_mode={load_mode}, peak_vram_gb={peak_vram_gb}")
            for dataset_name, strategy in pending:
                run_new_condition(model_name, cfg, model, tokenizer, peak_vram_gb, load_mode, dataset_name, strategy)
        except Exception as e:
            any_errors = True
            log(f"MODEL_LEVEL_ERROR {model_name}: {type(e).__name__}: {e}")
            log(traceback.format_exc())
        finally:
            try:
                if model is not None:
                    unload_model(model)
            except Exception:
                pass

    log("\n[3] Summarizing ablation outputs...")
    summary, pair_df, agg = summarize_outputs()

    expected_rows = len(MODELS) * len(DATASETS) * len(ALL_ABLATION_STRATEGIES)
    complete_rows = int(((summary["present"] == 1) & (summary["n"] >= LIMIT)).sum())
    error_rows = int(summary.get("error_rows", pd.Series([0])).sum())

    log("")
    log("============================================================")
    log("MC PROMPT ABLATION PREVIEW")
    log("============================================================")
    log("")
    log("Grouped summary:")
    print(agg.to_string(index=False))
    log("")
    log("Pairwise comparison:")
    print(pair_df.to_string(index=False))
    log("")
    log(f"Complete condition rows: {complete_rows}/{expected_rows}")
    log(f"Total error rows: {error_rows}")
    log(f"Saved outputs in: {OUT}")
    log("")

    if complete_rows == expected_rows and error_rows == 0 and not any_errors:
        log("MC PROMPT ABLATION OK")
    elif complete_rows == expected_rows:
        log("MC PROMPT ABLATION COMPLETED WITH ERRORS")
    else:
        log("MC PROMPT ABLATION INCOMPLETE")

if __name__ == "__main__":
    main()
