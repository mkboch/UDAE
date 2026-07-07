from pathlib import Path
import pandas as pd
import numpy as np
import re, math

ROOT = Path(".")
TAG = "mc_prompt_ablation_v3_full500"
OUT = Path("review_response_runs/mc_prompt_ablation_v3_full500")
POST = OUT / "posthoc_summary"
RAW = Path("results/raw")
POST.mkdir(parents=True, exist_ok=True)

MODELS = ["gemma_4_26b_a4b", "gemma_4_e4b", "qwen3_8b", "phi_4_reasoning"]
DATASETS = ["arc_challenge", "truthfulqa_mc1"]
BASELINE_STRATEGIES = [
    "orig_zero_shot_answer_only",
    "orig_cot_answer_only",
    "orig_few_shot_cot_answer_only",
]
NEW_STRATEGIES = [
    "cot_rationale_final_letter",
    "few_shot_cot_rationale_final_letter",
]
ALL_STRATEGIES = BASELINE_STRATEGIES + NEW_STRATEGIES

def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    df2 = df.copy()
    for c in df2.columns:
        df2[c] = df2[c].map(lambda x: "" if pd.isna(x) else x)
    cols = list(df2.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join([":---" for _ in cols]) + " |")
    for _, r in df2.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)

def save(df, name):
    df.to_csv(POST / f"{name}.csv", index=False)
    (POST / f"{name}.md").write_text(md_table(df))

def read_condition(model, dataset, strategy):
    p = RAW / f"{TAG}__{model}__{dataset}__{strategy}__n500.csv"
    if not p.exists():
        return None, p
    try:
        df = pd.read_csv(p)
        return df, p
    except Exception:
        return None, p

def as_num(s, default=0):
    return pd.to_numeric(s, errors="coerce").fillna(default)

def missing_rate(series):
    return float(series.fillna("").astype(str).str.strip().eq("").mean())

def contains_rate(series, pat):
    return float(series.fillna("").astype(str).str.contains(pat, regex=True, flags=re.I).mean())

def bare_letter_rate(series):
    return float(series.fillna("").astype(str).map(lambda x: bool(re.match(r"^\s*[A-E]\s*$", x, flags=re.I))).mean())

rows = []
for model in MODELS:
    for dataset in DATASETS:
        for strategy in ALL_STRATEGIES:
            df, p = read_condition(model, dataset, strategy)
            if df is None:
                rows.append({
                    "model": model, "dataset": dataset, "strategy": strategy,
                    "present": 0, "n": 0, "status": "MISSING", "file": str(p)
                })
                continue

            pred = df["prediction"] if "prediction" in df.columns else pd.Series([""] * len(df))
            resp = df["response_text"] if "response_text" in df.columns else pd.Series([""] * len(df))
            strict = as_num(df["correct"]) if "correct" in df.columns else pd.Series([0] * len(df))
            lenient = as_num(df["diagnostic_lenient_correct"]) if "diagnostic_lenient_correct" in df.columns else pd.Series([np.nan] * len(df))
            err = df["error"] if "error" in df.columns else pd.Series([""] * len(df))
            lat = as_num(df["latency_sec"], default=np.nan) if "latency_sec" in df.columns else pd.Series([np.nan] * len(df))
            tok = as_num(df["n_output_tokens"], default=np.nan) if "n_output_tokens" in df.columns else pd.Series([np.nan] * len(df))

            rows.append({
                "model": model,
                "dataset": dataset,
                "strategy": strategy,
                "present": 1,
                "n": len(df),
                "strict_acc": float(strict.mean()),
                "diagnostic_lenient_acc": float(lenient.mean()) if not lenient.isna().all() else np.nan,
                "missing_prediction_rate": missing_rate(pred),
                "final_hash_answer_rate": contains_rate(resp, r"####\s*[A-E]\b"),
                "bare_letter_response_rate": bare_letter_rate(resp),
                "think_tag_rate": contains_rate(resp, r"<\s*/?\s*think\s*>"),
                "object_object_rate": contains_rate(resp, r"\[object Object\]"),
                "median_response_chars": float(resp.fillna("").astype(str).str.len().median()),
                "mean_response_chars": float(resp.fillna("").astype(str).str.len().mean()),
                "mean_latency_sec": float(lat.mean()),
                "mean_output_tokens": float(tok.mean()),
                "error_rows": int(err.fillna("").astype(str).str.strip().ne("").sum()),
                "load_modes": ",".join(sorted(set(df["load_mode"].dropna().astype(str)))) if "load_mode" in df.columns else "",
                "status": "PASS" if len(df) == 500 else "BAD_N",
                "file": str(p),
            })

summary = pd.DataFrame(rows)
save(summary, "v3_full500_condition_summary")

grouped = summary.groupby(["dataset", "strategy"], dropna=False).agg(
    mean_strict_acc=("strict_acc", "mean"),
    mean_diagnostic_lenient_acc=("diagnostic_lenient_acc", "mean"),
    mean_missing_prediction_rate=("missing_prediction_rate", "mean"),
    mean_final_hash_answer_rate=("final_hash_answer_rate", "mean"),
    mean_bare_letter_response_rate=("bare_letter_response_rate", "mean"),
    mean_think_tag_rate=("think_tag_rate", "mean"),
    mean_latency_sec=("mean_latency_sec", "mean"),
    total_error_rows=("error_rows", "sum"),
).reset_index()
save(grouped, "v3_full500_grouped_summary")

pair_rows = []
pairs = [
    ("orig_cot_answer_only", "cot_rationale_final_letter"),
    ("orig_few_shot_cot_answer_only", "few_shot_cot_rationale_final_letter"),
]
for model in MODELS:
    for dataset in DATASETS:
        sub = summary[(summary.model == model) & (summary.dataset == dataset)]
        for old, new in pairs:
            a = sub[sub.strategy == old]
            b = sub[sub.strategy == new]
            if a.empty or b.empty:
                continue
            a = a.iloc[0]
            b = b.iloc[0]
            pair_rows.append({
                "model": model,
                "dataset": dataset,
                "comparison": f"{old} -> {new}",
                "old_strict_acc": a["strict_acc"],
                "new_strict_finalline_acc": b["strict_acc"],
                "delta_new_minus_old_strict": b["strict_acc"] - a["strict_acc"],
                "old_lenient_acc": a["diagnostic_lenient_acc"],
                "new_lenient_acc": b["diagnostic_lenient_acc"],
                "delta_new_minus_old_lenient": b["diagnostic_lenient_acc"] - a["diagnostic_lenient_acc"],
                "old_missing_prediction_rate": a["missing_prediction_rate"],
                "new_missing_prediction_rate": b["missing_prediction_rate"],
                "new_final_hash_answer_rate": b["final_hash_answer_rate"],
                "old_bare_letter_response_rate": a["bare_letter_response_rate"],
                "new_bare_letter_response_rate": b["bare_letter_response_rate"],
                "old_think_tag_rate": a["think_tag_rate"],
                "new_think_tag_rate": b["think_tag_rate"],
                "old_mean_latency_sec": a["mean_latency_sec"],
                "new_mean_latency_sec": b["mean_latency_sec"],
            })

pairs_df = pd.DataFrame(pair_rows)
save(pairs_df, "v3_full500_pairwise_prompt_comparison")

# Compact reviewer-facing table
reviewer_table = pairs_df.copy()
for c in [
    "old_strict_acc", "new_strict_finalline_acc", "delta_new_minus_old_strict",
    "old_lenient_acc", "new_lenient_acc", "delta_new_minus_old_lenient",
    "new_final_hash_answer_rate", "new_missing_prediction_rate"
]:
    if c in reviewer_table.columns:
        reviewer_table[c] = reviewer_table[c].astype(float).round(3)
save(reviewer_table, "v3_full500_reviewer_prompt_ablation_table")

# Validation checklist
valid = []
valid.append({
    "check": "All 16 new full-500 rationale conditions present",
    "status": "PASS" if (summary[summary.strategy.isin(NEW_STRATEGIES)]["n"].eq(500).all() and len(summary[summary.strategy.isin(NEW_STRATEGIES)]) == 16) else "FAIL"
})
valid.append({
    "check": "All 24 copied full-500 baseline conditions present",
    "status": "PASS" if (summary[summary.strategy.isin(BASELINE_STRATEGIES)]["n"].eq(500).all() and len(summary[summary.strategy.isin(BASELINE_STRATEGIES)]) == 24) else "FAIL"
})
valid.append({
    "check": "No model inference error rows",
    "status": "PASS" if int(summary["error_rows"].sum()) == 0 else "FAIL"
})
valid.append({
    "check": "New prompt load mode is bf16 direct/no 4-bit",
    "status": "PASS" if all("bf16_direct_no_4bit" in x for x in summary[summary.strategy.isin(NEW_STRATEGIES)]["load_modes"].astype(str)) else "FAIL"
})
valid.append({
    "check": "V3 full-500, not 238",
    "status": "PASS" if summary["n"].min() == 500 and summary["n"].max() == 500 else "FAIL"
})
valid_df = pd.DataFrame(valid)
save(valid_df, "v3_full500_validation_checklist")

status_lines = []
status_lines.append("# MC Prompt Ablation V3 Full-500 Status")
status_lines.append("")
status_lines.append("## Validation")
status_lines.append("")
status_lines.append(md_table(valid_df))
status_lines.append("")
status_lines.append("## Reviewer-facing prompt-ablation table")
status_lines.append("")
status_lines.append(md_table(reviewer_table))
status_lines.append("")
status_lines.append("## Grouped summary")
status_lines.append("")
status_lines.append(md_table(grouped))
status_lines.append("")
status_lines.append("## Interpretation")
status_lines.append("")
status_lines.append(
    "The V3 ablation uses the full 500 ARC-Challenge and TruthfulQA MC1 examples from journal_full500_v1, "
    "not the earlier 238-example matched subset. It compares the original answer-only multiple-choice protocol "
    "against rationale-allowed prompts requiring a final `#### <letter>` answer line. All new inference runs use "
    "`bf16_direct_no_4bit`, avoiding quantization ambiguity."
)
status_lines.append("")
status_lines.append("## Files")
for f in sorted(POST.glob("*.csv")):
    status_lines.append(f"- `{f}`")
for f in sorted(POST.glob("*.md")):
    status_lines.append(f"- `{f}`")

(OUT / "MC_PROMPT_ABLATION_V3_FULL500_STATUS.md").write_text("\n".join(status_lines))
(POST / "MC_PROMPT_ABLATION_V3_FULL500_STATUS.md").write_text("\n".join(status_lines))

print("============================================================")
print("V3 FULL500 POSTHOC SUMMARY")
print("============================================================")
print(valid_df.to_string(index=False))
print()
print("Reviewer table:")
print(reviewer_table.to_string(index=False))
print()
print("Grouped summary:")
print(grouped.to_string(index=False))
print()
if (valid_df["status"] == "PASS").all():
    print("V3 FULL500 SUMMARY OK")
else:
    print("V3 FULL500 SUMMARY HAS FAILURES")
print(f"Saved: {OUT / 'MC_PROMPT_ABLATION_V3_FULL500_STATUS.md'}")
