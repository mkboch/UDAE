from pathlib import Path
import re
import pandas as pd

OUT = Path("review_response_runs/full_run_audit_v1")
OUT.mkdir(parents=True, exist_ok=True)

RAW = Path("results/raw")
TAG = "journal_full500_v1"

MODELS = [
    "gemma_4_26b_a4b",
    "gemma_4_e2b",
    "gemma_4_e4b",
    "phi_4_mini_reasoning",
    "phi_4_reasoning",
    "qwen3_30b_a3b",
    "qwen3_8b",
]
FOCUS_MODELS = ["gemma_4_26b_a4b", "gemma_4_e4b", "qwen3_8b", "phi_4_reasoning"]
DATASETS = ["arc_challenge", "gsm8k", "math_l1_l3", "truthfulqa_mc1"]
MC_DATASETS = ["arc_challenge", "truthfulqa_mc1"]
STRATEGIES = ["zero_shot", "cot", "few_shot_cot"]

def find_file(tag, model, dataset, strategy):
    files = sorted(RAW.glob(f"{tag}__{model}__{dataset}__{strategy}__n*.csv"))
    if not files:
        return None
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[0]

rows = []
for m in MODELS:
    for d in DATASETS:
        for s in STRATEGIES:
            p = find_file(TAG, m, d, s)
            if p is None:
                rows.append({
                    "tag": TAG, "model": m, "dataset": d, "strategy": s,
                    "present": 0, "n_from_filename": 0, "n_rows": 0, "file": ""
                })
                continue
            m_n = re.search(r"__n(\d+)\.csv$", p.name)
            n_file = int(m_n.group(1)) if m_n else -1
            try:
                n_rows = len(pd.read_csv(p))
            except Exception:
                n_rows = -1
            rows.append({
                "tag": TAG, "model": m, "dataset": d, "strategy": s,
                "present": 1, "n_from_filename": n_file, "n_rows": n_rows, "file": str(p)
            })

df = pd.DataFrame(rows)
df.to_csv(OUT / "journal_full500_v1_inventory.csv", index=False)
(OUT / "journal_full500_v1_inventory.md").write_text(df.to_markdown(index=False))

dataset_summary = df.groupby("dataset").agg(
    conditions_present=("present", "sum"),
    min_rows=("n_rows", "min"),
    max_rows=("n_rows", "max"),
).reset_index()

expected = []
for d in DATASETS:
    if d == "math_l1_l3":
        expected_n = 238
    else:
        expected_n = 500
    sub = df[df["dataset"] == d]
    ok = (
        len(sub) == 21
        and int(sub["present"].sum()) == 21
        and int(sub["n_rows"].min()) >= expected_n
        and int(sub["n_from_filename"].min()) >= expected_n
    )
    expected.append({
        "dataset": d,
        "expected_conditions": 21,
        "expected_n": expected_n,
        "conditions_present": int(sub["present"].sum()),
        "min_n_from_filename": int(sub["n_from_filename"].min()),
        "max_n_from_filename": int(sub["n_from_filename"].max()),
        "min_n_rows": int(sub["n_rows"].min()),
        "max_n_rows": int(sub["n_rows"].max()),
        "status": "PASS" if ok else "FAIL",
    })

summary = pd.DataFrame(expected)
summary.to_csv(OUT / "journal_full500_v1_dataset_summary.csv", index=False)
(OUT / "journal_full500_v1_dataset_summary.md").write_text(summary.to_markdown(index=False))

# Check whether full-500 MC ablation baselines exist for the representative models.
base_rows = []
for m in FOCUS_MODELS:
    for d in MC_DATASETS:
        for s in STRATEGIES:
            p = find_file(TAG, m, d, s)
            n_rows = len(pd.read_csv(p)) if p and p.exists() else 0
            base_rows.append({
                "model": m,
                "dataset": d,
                "baseline_strategy": s,
                "required_for_full500_prompt_ablation": 1,
                "present": int(p is not None),
                "n_rows": n_rows,
                "status": "PASS" if p is not None and n_rows >= 500 else "FAIL",
                "file": str(p) if p else "",
            })

base = pd.DataFrame(base_rows)
base.to_csv(OUT / "full500_mc_ablation_baseline_check.csv", index=False)
(OUT / "full500_mc_ablation_baseline_check.md").write_text(base.to_markdown(index=False))

all_full_ok = (summary["status"] == "PASS").all()
mc_baselines_ok = (base["status"] == "PASS").all()

decision = []
decision.append("# Full Run Audit Decision")
decision.append("")
decision.append("## journal_full500_v1 dataset coverage")
decision.append("")
decision.append(summary.to_markdown(index=False))
decision.append("")
decision.append("## Full-500 MC prompt-ablation baseline availability")
decision.append("")
decision.append(base.to_markdown(index=False))
decision.append("")
decision.append("## Decision")
decision.append("")
if all_full_ok and mc_baselines_ok:
    decision.append("PASS: The larger/full run exists for ARC-Challenge, GSM8K, and TruthfulQA MC1 at n=500, with MATH L1-L3 retained at n=238. Full-500 MC prompt ablation can be run for ARC-Challenge and TruthfulQA MC1.")
else:
    decision.append("FAIL: Full run or required MC baselines are incomplete. Do not start full-500 prompt ablation until missing files are resolved.")
decision.append("")
decision.append(f"- all_full_ok: {all_full_ok}")
decision.append(f"- mc_baselines_ok: {mc_baselines_ok}")

(OUT / "FULL_RUN_AUDIT_DECISION.md").write_text("\n".join(decision))

print("============================================================")
print("journal_full500_v1 dataset summary")
print("============================================================")
print(summary.to_string(index=False))
print()
print("============================================================")
print("full500 MC ablation baseline check")
print("============================================================")
print(base.to_string(index=False))
print()
print("============================================================")
if all_full_ok and mc_baselines_ok:
    print("FULL RUN AUDIT PASS")
    print("Recommendation: use full-500 MC prompt ablation, not the stopped 238 V2 ablation.")
else:
    print("FULL RUN AUDIT FAIL")
    print("Recommendation: fix missing full-run files before prompt ablation.")
print("============================================================")
