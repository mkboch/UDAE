from pathlib import Path
from collections import Counter, defaultdict
import re, json, math, subprocess, sys
import numpy as np
import pandas as pd

ROOT = Path(".")
OUT = Path("review_response_runs/experiments_1_to_5_check_v1")
OUT.mkdir(parents=True, exist_ok=True)

ORIG_TAG = "workshop_unified238_v1"
EXT_TAG = "journal_full500_v1"

MODELS = [
    "gemma_4_26b_a4b",
    "gemma_4_e2b",
    "gemma_4_e4b",
    "phi_4_mini_reasoning",
    "phi_4_reasoning",
    "qwen3_30b_a3b",
    "qwen3_8b",
]
DATASETS = ["arc_challenge", "gsm8k", "math_l1_l3", "truthfulqa_mc1"]
STRATEGIES = ["zero_shot", "cot", "few_shot_cot"]
WEIGHTS = {"arc_challenge": 0.20, "gsm8k": 0.40, "math_l1_l3": 0.30, "truthfulqa_mc1": 0.10}

def log(x=""):
    print(x, flush=True)

def md_table(df):
    return df.to_markdown(index=False)

def save_table(df, stem):
    df.to_csv(OUT / f"{stem}.csv", index=False)
    (OUT / f"{stem}.md").write_text(md_table(df))

def find_file(tag, model, dataset, strategy):
    files = sorted(Path("results/raw").glob(f"{tag}__{model}__{dataset}__{strategy}__n*.csv"))
    if not files:
        return None
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[0]

def read_condition(tag, model, dataset, strategy):
    p = find_file(tag, model, dataset, strategy)
    if p is None:
        return None
    df = pd.read_csv(p)
    df["__file"] = str(p)
    df["__tag"] = tag
    df["__model"] = model
    df["__dataset"] = dataset
    df["__strategy"] = strategy
    return df

def condition_inventory(tag):
    rows = []
    for m in MODELS:
        for d in DATASETS:
            for s in STRATEGIES:
                p = find_file(tag, m, d, s)
                if p is None:
                    rows.append({"tag": tag, "model": m, "dataset": d, "strategy": s, "present": 0, "n": 0, "file": ""})
                else:
                    n_match = re.search(r"__n(\d+)\.csv$", p.name)
                    n = int(n_match.group(1)) if n_match else -1
                    rows.append({"tag": tag, "model": m, "dataset": d, "strategy": s, "present": 1, "n": n, "file": str(p)})
    return pd.DataFrame(rows)

def strict_acc(df):
    return float(pd.to_numeric(df["correct"], errors="coerce").fillna(0).mean())

def weighted_scores(tag):
    rows = []
    for m in MODELS:
        for s in STRATEGIES:
            ok = True
            score = 0.0
            lat = []
            vram = []
            for d, w in WEIGHTS.items():
                df = read_condition(tag, m, d, s)
                if df is None:
                    ok = False
                    break
                score += w * strict_acc(df)
                lat.append(float(pd.to_numeric(df.get("latency_sec", pd.Series(dtype=float)), errors="coerce").mean()))
                vram.append(float(pd.to_numeric(df.get("peak_vram_gb", pd.Series(dtype=float)), errors="coerce").max()))
            if ok:
                rows.append({
                    "tag": tag,
                    "model": m,
                    "strategy": s,
                    "weighted_acc": score,
                    "mean_latency_sec": float(np.nanmean(lat)),
                    "max_peak_vram_gb": float(np.nanmax(vram)),
                })
    out = pd.DataFrame(rows)
    if len(out):
        out["rank"] = out["weighted_acc"].rank(method="min", ascending=False).astype(int)
        out = out.sort_values(["rank", "model", "strategy"])
    return out

def is_missing(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return True
    return str(x).strip().lower() in {"", "nan", "none", "null"}

def has_think(x):
    return bool(re.search(r"<\s*/?\s*think\s*>", str(x), flags=re.I))

def has_object_object(x):
    return "[object Object]" in str(x)

def strip_traces(x):
    s = "" if x is None or (isinstance(x, float) and math.isnan(x)) else str(x)
    s = s.replace("[object Object]", " ")
    s = re.sub(r"<think>.*?</think>", " ", s, flags=re.I | re.S)
    s = re.sub(r"<\s*/?\s*think\s*>", " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_mc_lenient(text):
    s = strip_traces(text)
    pats = [
        r"####\s*([A-E])\b",
        r"(?:final\s+answer|answer|option|choice)\s*(?:is|:)?\s*([A-E])\b",
        r"\b([A-E])\b",
    ]
    for pat in pats:
        hits = re.findall(pat, s, flags=re.I)
        if hits:
            return hits[-1].upper()
    return ""

def parse_math_lenient(text):
    s = strip_traces(text)
    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", s)
    if boxed:
        return boxed[-1].strip()
    hashes = re.findall(r"####\s*([^\n\r]+)", s)
    if hashes:
        return hashes[-1].strip()
    hits = re.findall(r"(?:final\s+answer|answer)\s*(?:is|:)?\s*([-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?)", s, flags=re.I)
    if hits:
        return hits[-1].strip()
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?", s)
    return nums[-1].strip() if nums else ""

def normalize_mc(x):
    m = re.search(r"\b([A-E])\b", str(x).upper())
    return m.group(1) if m else ""

def parse_number(x):
    s = str(x).replace(",", "")
    frac = re.search(r"([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)", s)
    if frac:
        try:
            den = float(frac.group(2))
            if den != 0:
                return float(frac.group(1)) / den
        except Exception:
            pass
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if nums:
        try:
            return float(nums[-1])
        except Exception:
            return None
    return None

def math_equal(pred, gold):
    pv, gv = parse_number(pred), parse_number(gold)
    if pv is not None and gv is not None:
        return abs(pv - gv) <= max(1e-6, 1e-6 * abs(gv))
    return str(pred).strip().lower() == str(gold).strip().lower()

def diagnostic_lenient_correct(row):
    d = row["__dataset"]
    if d in ["arc_challenge", "truthfulqa_mc1"]:
        pred = parse_mc_lenient(row.get("response_text", ""))
        return int(pred != "" and pred == normalize_mc(row.get("gold_answer", ""))), pred
    pred = parse_math_lenient(row.get("response_text", ""))
    return int(pred != "" and math_equal(pred, row.get("gold_answer", ""))), pred

def holm_adjust(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [None] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj

def paired_permutation(config_a, config_b, B=10000, seed=123):
    rng = np.random.default_rng(seed)
    diffs_by_dataset = {}
    obs = 0.0
    for d, w in WEIGHTS.items():
        df_a = read_condition(ORIG_TAG, config_a[0], d, config_a[1])
        df_b = read_condition(ORIG_TAG, config_b[0], d, config_b[1])
        a = df_a[["sample_id", "correct"]].copy()
        b = df_b[["sample_id", "correct"]].copy()
        a["sample_id"] = a["sample_id"].astype(str)
        b["sample_id"] = b["sample_id"].astype(str)
        a["correct"] = pd.to_numeric(a["correct"], errors="coerce").fillna(0).astype(float)
        b["correct"] = pd.to_numeric(b["correct"], errors="coerce").fillna(0).astype(float)
        merged = a.merge(b, on="sample_id", suffixes=("_a", "_b"))
        diff = (merged["correct_a"] - merged["correct_b"]).to_numpy(dtype=float)
        diffs_by_dataset[d] = diff
        obs += w * float(diff.mean())

    perm = np.zeros(B, dtype=float)
    for d, diff in diffs_by_dataset.items():
        signs = rng.choice(np.array([-1.0, 1.0]), size=(B, len(diff)))
        perm += WEIGHTS[d] * (signs * diff).mean(axis=1)

    p = float((np.sum(np.abs(perm) >= abs(obs)) + 1) / (B + 1))
    return obs, p

log("============================================================")
log("CHECK EXPERIMENTS 1 TO 5 FOR TMLR REVIEW RESPONSE")
log("============================================================")
log(f"PWD: {Path.cwd()}")
log(f"Python: {sys.version}")
log("")

# ------------------------------------------------------------------
# Exp 1: larger/full-size robustness availability
# ------------------------------------------------------------------
log("[EXP 1] Checking larger/full-size robustness coverage...")
inv_orig = condition_inventory(ORIG_TAG)
inv_ext = condition_inventory(EXT_TAG)
save_table(inv_orig, "exp1_original_n238_inventory")
save_table(inv_ext, "exp1_extended_inventory")

coverage_rows = []
for d in DATASETS:
    sub_orig = inv_orig[inv_orig["dataset"] == d]
    sub_ext = inv_ext[inv_ext["dataset"] == d]
    coverage_rows.append({
        "dataset": d,
        "submitted_conditions_present": int(sub_orig["present"].sum()),
        "submitted_n_values": ",".join(map(str, sorted(sub_orig[sub_orig["present"] == 1]["n"].unique()))),
        "extended_conditions_present": int(sub_ext["present"].sum()),
        "extended_n_values": ",".join(map(str, sorted(sub_ext[sub_ext["present"] == 1]["n"].unique()))),
        "status": "PASS" if int(sub_ext["present"].sum()) == 21 else "MISSING",
    })
coverage = pd.DataFrame(coverage_rows)
save_table(coverage, "exp1_larger_size_coverage_by_dataset")

orig_ws = weighted_scores(ORIG_TAG)
ext_ws = weighted_scores(EXT_TAG)
save_table(orig_ws, "exp1_original_weighted_scores")
save_table(ext_ws, "exp1_extended_weighted_scores")

comparison = orig_ws[["model", "strategy", "rank", "weighted_acc"]].rename(columns={"rank": "submitted_rank", "weighted_acc": "submitted_weighted_acc"}).merge(
    ext_ws[["model", "strategy", "rank", "weighted_acc"]].rename(columns={"rank": "extended_rank", "weighted_acc": "extended_weighted_acc"}),
    on=["model", "strategy"],
    how="outer",
)
comparison["delta_extended_minus_submitted"] = comparison["extended_weighted_acc"] - comparison["submitted_weighted_acc"]
comparison = comparison.sort_values(["submitted_rank", "extended_rank"])
save_table(comparison, "exp1_original_vs_extended_weighted_comparison")

# ------------------------------------------------------------------
# Exp 2: repeated subset stability
# ------------------------------------------------------------------
log("[EXP 2] Checking repeated-subset / bootstrap stability outputs...")
prev_t2 = Path("review_response_runs/final_tables_v1/table2_rank_stability_original_and_extended.csv")
if prev_t2.exists():
    t2 = pd.read_csv(prev_t2)
    save_table(t2, "exp2_rank_stability_original_and_extended")
    exp2_status = "PASS_EXISTING_TABLE_FOUND"
else:
    t2 = pd.DataFrame()
    exp2_status = "MISSING_FINAL_TABLE_RUN_ANALYSIS_V1_FIRST"

# ------------------------------------------------------------------
# Exp 3: parser-aware Phi diagnostic with artifact rates
# ------------------------------------------------------------------
log("[EXP 3] Computing Phi parser/artifact diagnostics...")
phi_rows = []
all_parser_rows = []
for m in MODELS:
    for d in DATASETS:
        for s in STRATEGIES:
            df = read_condition(ORIG_TAG, m, d, s)
            if df is None:
                continue
            df = df.copy()
            df["correct_num"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)
            df["strict_missing"] = df["prediction"].apply(is_missing)
            df["think_tag"] = df["response_text"].apply(has_think)
            df["object_object"] = df["response_text"].apply(has_object_object)
            lenient = df.apply(diagnostic_lenient_correct, axis=1)
            df["diagnostic_lenient_correct"] = [x[0] for x in lenient]
            df["diagnostic_lenient_prediction"] = [x[1] for x in lenient]
            terminal_hash = df["response_text"].astype(str).str.contains(r"####\s*\S+\s*$", regex=True)
            all_parser_rows.append({
                "model": m,
                "dataset": d,
                "strategy": s,
                "n": len(df),
                "strict_acc": float(df["correct_num"].mean()),
                "strict_missing_prediction_rate": float(df["strict_missing"].mean()),
                "think_tag_rate": float(df["think_tag"].mean()),
                "object_object_rate": float(df["object_object"].mean()),
                "diagnostic_lenient_acc": float(df["diagnostic_lenient_correct"].mean()),
                "terminal_hash_answer_rate": float(terminal_hash.mean()) if d in ["gsm8k", "math_l1_l3"] else np.nan,
            })
            if m == "phi_4_reasoning":
                recovered = int(((df["strict_missing"]) & (df["diagnostic_lenient_correct"] == 1)).sum())
                phi_rows.append({
                    "model": m,
                    "dataset": d,
                    "strategy": s,
                    "n": len(df),
                    "strict_acc": float(df["correct_num"].mean()),
                    "strict_missing_prediction_rate": float(df["strict_missing"].mean()),
                    "think_tag_rate": float(df["think_tag"].mean()),
                    "object_object_rate": float(df["object_object"].mean()),
                    "diagnostic_lenient_acc": float(df["diagnostic_lenient_correct"].mean()),
                    "strict_parser_failures": int(df["strict_missing"].sum()),
                    "recovered_correct_from_strict_missing": recovered,
                    "terminal_hash_answer_rate": float(terminal_hash.mean()) if d in ["gsm8k", "math_l1_l3"] else np.nan,
                })

phi_diag = pd.DataFrame(phi_rows).sort_values(["dataset", "strategy"])
all_parser = pd.DataFrame(all_parser_rows).sort_values(["strict_missing_prediction_rate", "think_tag_rate"], ascending=False)
save_table(phi_diag, "exp3_phi4_parser_artifact_diagnostics")
save_table(all_parser.head(25), "exp3_top_parser_artifact_rates_all_conditions")

# ------------------------------------------------------------------
# Exp 4: MC prompt protocol diagnostic, no new GPU ablation
# ------------------------------------------------------------------
log("[EXP 4] Computing MC prompt-protocol diagnostics from existing outputs...")
mc_rows = []
letter_pat = re.compile(r"^\s*[A-E]\s*$", flags=re.I)
for d in ["arc_challenge", "truthfulqa_mc1"]:
    for m in MODELS:
        for s in STRATEGIES:
            df = read_condition(ORIG_TAG, m, d, s)
            if df is None:
                continue
            resp = df["response_text"].fillna("").astype(str)
            pred = df["prediction"].fillna("")
            mc_rows.append({
                "dataset": d,
                "model": m,
                "strategy": s,
                "n": len(df),
                "strict_acc": strict_acc(df),
                "prediction_missing_rate": float(pred.apply(is_missing).mean()),
                "bare_letter_response_rate": float(resp.apply(lambda x: bool(letter_pat.match(x))).mean()),
                "median_response_chars": float(resp.str.len().median()),
                "mean_response_chars": float(resp.str.len().mean()),
                "think_tag_rate": float(resp.apply(has_think).mean()),
                "object_object_rate": float(resp.apply(has_object_object).mean()),
            })
mc_diag = pd.DataFrame(mc_rows).sort_values(["dataset", "strategy", "model"])
save_table(mc_diag, "exp4_mc_prompt_protocol_response_diagnostics")

# Math-only prompt sensitivity, to support revised/limited prompt claim
math_rows = []
for m in MODELS:
    for s in STRATEGIES:
        vals = []
        for d in ["gsm8k", "math_l1_l3"]:
            df = read_condition(ORIG_TAG, m, d, s)
            if df is not None:
                vals.append(strict_acc(df))
        if len(vals) == 2:
            math_rows.append({"model": m, "strategy": s, "math_only_mean_acc": float(np.mean(vals))})
math_df = pd.DataFrame(math_rows)
if len(math_df):
    math_df["rank"] = math_df.groupby("strategy")["math_only_mean_acc"].rank(method="min", ascending=False).astype(int)
    rank_range = math_df.groupby("model")["rank"].agg(best_rank="min", worst_rank="max").reset_index()
    rank_range["math_only_rank_range"] = rank_range["worst_rank"] - rank_range["best_rank"]
    save_table(math_df.sort_values(["strategy", "rank"]), "exp4_math_only_prompt_sensitivity_by_strategy")
    save_table(rank_range.sort_values("math_only_rank_range", ascending=False), "exp4_math_only_prompt_rank_range")

ablation_files = list(Path("results/raw").glob("*ablation*")) + list(Path("results/raw").glob("*promptfix*")) + list(Path("results/raw").glob("*mc_cot*"))
ablation_status = "NO_NEW_GPU_ABLATION_FOUND_TEXTUAL_REFRAMING_NEEDED" if not ablation_files else "EXISTING_ABLATION_FILES_FOUND"

# ------------------------------------------------------------------
# Exp 5: Holm-corrected paired permutation tests
# ------------------------------------------------------------------
log("[EXP 5] Running Holm-corrected paired permutation tests for top comparisons...")
comparisons = [
    (("gemma_4_26b_a4b", "zero_shot"), ("gemma_4_e4b", "few_shot_cot")),
    (("gemma_4_26b_a4b", "zero_shot"), ("gemma_4_e4b", "cot")),
    (("gemma_4_26b_a4b", "zero_shot"), ("gemma_4_e4b", "zero_shot")),
    (("gemma_4_26b_a4b", "zero_shot"), ("gemma_4_26b_a4b", "cot")),
    (("gemma_4_e4b", "few_shot_cot"), ("gemma_4_e4b", "cot")),
    (("gemma_4_e4b", "few_shot_cot"), ("gemma_4_e4b", "zero_shot")),
    (("gemma_4_e4b", "cot"), ("gemma_4_e4b", "zero_shot")),
]
perm_rows = []
for a, b in comparisons:
    delta, p = paired_permutation(a, b, B=10000, seed=123)
    perm_rows.append({
        "model_A": a[0],
        "strategy_A": a[1],
        "model_B": b[0],
        "strategy_B": b[1],
        "weighted_delta_A_minus_B": delta,
        "raw_two_sided_p": p,
    })
perm = pd.DataFrame(perm_rows)
perm["holm_adjusted_p"] = holm_adjust(list(perm["raw_two_sided_p"]))
perm["holm_significant_at_0.05"] = perm["holm_adjusted_p"] < 0.05
save_table(perm, "exp5_holm_corrected_paired_permutation_tests")

# ------------------------------------------------------------------
# Final checklist
# ------------------------------------------------------------------
log("Writing final checklist...")
checklist = []

ext_nonmath_ok = True
for d in ["arc_challenge", "gsm8k", "truthfulqa_mc1"]:
    sub = inv_ext[(inv_ext["dataset"] == d) & (inv_ext["present"] == 1)]
    if len(sub) != 21 or set(sub["n"].unique()) != {500}:
        ext_nonmath_ok = False
math_ok = len(inv_ext[(inv_ext["dataset"] == "math_l1_l3") & (inv_ext["present"] == 1)]) == 21

checklist.append({
    "experiment": "1 larger/full-size robustness beyond submitted n=238",
    "status": "PASS" if ext_nonmath_ok and math_ok else "PARTIAL_OR_MISSING",
    "evidence_file": "exp1_larger_size_coverage_by_dataset.md; exp1_original_vs_extended_weighted_comparison.md",
    "paper_action": "Add larger non-MATH robustness analysis; explain MATH L1-L3 matched cap."
})
checklist.append({
    "experiment": "2 repeated-subset / bootstrap rank stability",
    "status": exp2_status,
    "evidence_file": "exp2_rank_stability_original_and_extended.md",
    "paper_action": "Add winner frequency, interval, variance/stability table."
})
checklist.append({
    "experiment": "3 Phi parser-aware diagnostic",
    "status": "PASS",
    "evidence_file": "exp3_phi4_parser_artifact_diagnostics.md",
    "paper_action": "Separate parser failures from wrong answers; revise Phi claim."
})
checklist.append({
    "experiment": "4 MC prompt protocol diagnostic",
    "status": ablation_status,
    "evidence_file": "exp4_mc_prompt_protocol_response_diagnostics.md; exp4_math_only_prompt_rank_range.md",
    "paper_action": "Clarify MC CoT was answer-format-constrained; weaken prompt-sensitivity claims or run optional GPU ablation later."
})
checklist.append({
    "experiment": "5 bootstrap/permutation procedure + multiple comparisons",
    "status": "PASS",
    "evidence_file": "exp5_holm_corrected_paired_permutation_tests.md",
    "paper_action": "Explain paired sign-flip permutation, resampling unit, and Holm correction."
})
checklist_df = pd.DataFrame(checklist)
save_table(checklist_df, "EXPERIMENTS_1_TO_5_CHECKLIST")

summary = []
summary.append("# Experiments 1–5 Check Summary")
summary.append("")
summary.append("This check excludes experiment 6, the reproducibility package audit, which can be completed later before final revision/upload.")
summary.append("")
summary.append("## Checklist")
summary.append("")
summary.append(md_table(checklist_df))
summary.append("")
summary.append("## Key outputs")
for p in sorted(OUT.glob("*.md")):
    summary.append(f"- `{p}`")
summary.append("")
summary.append("## Main interpretation")
summary.append("")
summary.append("- Larger-size robustness: use the extended non-MATH n=500 results and repeated 238-subset analysis to address the one-seed/n=238 concern.")
summary.append("- Precision: already handled by prior precision table; all submitted conditions record bf16.")
summary.append("- Phi: revise from clean reasoning failure to parser/interface-compatibility diagnostic.")
summary.append("- Prompt protocol: current evidence supports a textual reframing; no new GPU prompt ablation was found in existing results.")
summary.append("- Statistics: use Holm-adjusted paired permutation table and clearly state the paired resampling/sign-flip design.")
(OUT / "EXPERIMENTS_1_TO_5_STATUS.md").write_text("\n".join(summary))

log("")
log("============================================================")
log("EXPERIMENT CHECK PREVIEW")
log("============================================================")
print(checklist_df.to_string(index=False))
log("")
log("Coverage by dataset:")
print(coverage.to_string(index=False))
log("")
log("Top original vs extended:")
print(comparison.head(10).to_string(index=False))
log("")
log("Phi parser/artifact diagnostics:")
print(phi_diag.to_string(index=False))
log("")
log("MC prompt protocol diagnostics, grouped preview:")
print(mc_diag.groupby(["dataset", "strategy"])[["strict_acc", "bare_letter_response_rate", "median_response_chars", "think_tag_rate"]].mean().reset_index().to_string(index=False))
log("")
log("Holm-corrected permutation tests:")
print(perm.to_string(index=False))
log("")
log("============================================================")
log("EXPERIMENTS 1 TO 5 CHECK OK")
log(f"Saved outputs in: {OUT}")
log("============================================================")
