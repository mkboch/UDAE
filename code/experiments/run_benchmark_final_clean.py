from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

from models.loader import load_model, unload_model
from models.inference import run_inference, tokens_per_second
from prompts.builder import build_prompt, maybe_apply_chat_template, get_generation_config
from evaluation.extractor import extract_answer
from evaluation.grader import grade_prediction

ROOT = Path(__file__).resolve().parents[1]
MODELS_CFG = ROOT / "configs" / "models.yaml"
RAW_DIR = ROOT / "results" / "raw"

def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def load_dataset_records(dataset_name: str, limit: int) -> List[Dict[str, Any]]:
    path = ROOT / "data" / "prepared" / f"{dataset_name}.jsonl"
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows[:limit]

def model_map() -> Dict[str, Dict[str, Any]]:
    models = load_yaml(MODELS_CFG)["models"]
    return {m["name"]: m for m in models}

def write_rows_csv(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(out_path.parent)
    fieldnames = [
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
        "correct",
        "latency_sec",
        "n_output_tokens",
        "tokens_per_sec",
        "peak_vram_gb",
        "response_text",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--strategies", nargs="+", required=True)
    ap.add_argument("--limit", type=int, required=True)
    ap.add_argument("--max-new-tokens", type=int, required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    gen_cfg = get_generation_config()
    dataset_rows = load_dataset_records(args.dataset, args.limit)
    mmap = model_map()
    if args.model not in mmap:
        raise ValueError(f"Unknown model: {args.model}")

    chosen_cfg = mmap[args.model]
    model, tokenizer, peak_vram_gb, load_mode = load_model(chosen_cfg)

    try:
        all_rows = []
        for strategy in args.strategies:
            print(f"\n--- model={chosen_cfg['name']} dataset={args.dataset} strategy={strategy} n={len(dataset_rows)} ---")
            run_rows = []
            for i, rec in enumerate(dataset_rows, start=1):
                prompt = build_prompt(strategy, rec["question"], args.dataset)
                prompt = maybe_apply_chat_template(tokenizer, prompt)

                response_text, latency_sec, n_output_tokens = run_inference(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(gen_cfg["temperature"]),
                    do_sample=bool(gen_cfg["do_sample"]),
                )

                prediction = extract_answer(
                    response_text,
                    rec.get("extractor", None) or {
                        "gsm8k": "gsm8k",
                        "math_l1_l3": "math",
                        "arc_challenge": "arc",
                        "truthfulqa_mc1": "truthfulqa",
                    }[args.dataset],
                )
                correct = grade_prediction(prediction, rec["gold_answer"], args.dataset)
                tps = tokens_per_second(n_output_tokens, latency_sec)

                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "requested_model_name": chosen_cfg["name"],
                    "actual_model_name": chosen_cfg["name"],
                    "model_pretty_name": chosen_cfg["pretty_name"],
                    "hf_id": chosen_cfg["hf_id"],
                    "architecture": chosen_cfg["architecture"],
                    "total_params_b": chosen_cfg["total_params_b"],
                    "active_params_b": chosen_cfg["active_params_b"],
                    "load_mode": load_mode,
                    "dataset_name": args.dataset,
                    "strategy": strategy,
                    "sample_id": rec["sample_id"],
                    "question": rec["question"],
                    "gold_answer": rec["gold_answer"],
                    "prediction": prediction,
                    "correct": correct,
                    "latency_sec": latency_sec,
                    "n_output_tokens": n_output_tokens,
                    "tokens_per_sec": tps,
                    "peak_vram_gb": peak_vram_gb,
                    "response_text": response_text,
                }
                run_rows.append(row)
                all_rows.append(row)

                print(f"[{i}/{len(dataset_rows)}] sample_id={rec['sample_id']} correct={correct} latency={latency_sec:.3f}s tokens={n_output_tokens} tps={tps:.2f}")

            out_path = RAW_DIR / f"{args.tag}__{chosen_cfg['name']}__{args.dataset}__{strategy}__n{len(run_rows)}.csv"
            write_rows_csv(out_path, run_rows)
            print(f"Saved: {out_path}")

        combined = RAW_DIR / f"{args.tag}__{chosen_cfg['name']}__ALL_RUNS__n{len(all_rows)}.csv"
        write_rows_csv(combined, all_rows)
        print(f"Saved combined: {combined}")
    finally:
        unload_model(model)

if __name__ == "__main__":
    main()
