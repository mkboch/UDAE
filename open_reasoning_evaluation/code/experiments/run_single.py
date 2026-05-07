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
DATASETS_CFG = ROOT / "configs" / "datasets.yaml"
RAW_DIR = ROOT / "results" / "raw"


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prepared_records(dataset_name: str) -> List[Dict[str, Any]]:
    path = ROOT / "data" / "prepared" / f"{dataset_name}.jsonl"
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def select_model_cfg(model_name: str) -> Dict[str, Any]:
    cfg = load_yaml(MODELS_CFG)["models"]
    for m in cfg:
        if m["name"] == model_name:
            return m
    raise ValueError(f"Model not found in config: {model_name}")


def select_dataset_cfg(dataset_name: str) -> Dict[str, Any]:
    cfg = load_yaml(DATASETS_CFG)["datasets"]
    for d in cfg:
        if d["name"] == dataset_name:
            return d
    raise ValueError(f"Dataset not found in config: {dataset_name}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_rows_csv(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(out_path.parent)
    fieldnames = [
        "timestamp_utc",
        "model_name",
        "model_pretty_name",
        "hf_id",
        "architecture",
        "total_params_b",
        "active_params_b",
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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model config name")
    parser.add_argument("--dataset", required=True, help="Dataset config name")
    parser.add_argument("--strategy", required=True, choices=["zero_shot", "cot", "few_shot_cot"])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    args = parser.parse_args()

    model_cfg = select_model_cfg(args.model)
    dataset_cfg = select_dataset_cfg(args.dataset)
    gen_cfg = get_generation_config()

    records = load_prepared_records(args.dataset)
    records = records[: max(0, int(args.limit))]

    print(f"Loading model: {model_cfg['name']} ({model_cfg['hf_id']})")
    model, tokenizer, peak_vram_gb = load_model(model_cfg)
    print(f"Peak VRAM after load: {peak_vram_gb:.3f} GB")
    print(f"Running {len(records)} samples on dataset={args.dataset}, strategy={args.strategy}")

    rows: List[Dict[str, Any]] = []
    for i, rec in enumerate(records, start=1):
        prompt = build_prompt(args.strategy, rec["question"], args.dataset)
        prompt = maybe_apply_chat_template(tokenizer, prompt)
        response_text, latency_sec, n_output_tokens = run_inference(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=int(args.max_new_tokens) if int(args.max_new_tokens) > 0 else int(dataset_cfg["max_new_tokens"]),
            temperature=float(gen_cfg["temperature"]),
            do_sample=bool(gen_cfg["do_sample"]),
        )
        prediction = extract_answer(response_text, dataset_cfg["extractor"])
        correct = grade_prediction(prediction, rec["gold_answer"], args.dataset)
        tps = tokens_per_second(n_output_tokens, latency_sec)

        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": model_cfg["name"],
            "model_pretty_name": model_cfg["pretty_name"],
            "hf_id": model_cfg["hf_id"],
            "architecture": model_cfg["architecture"],
            "total_params_b": model_cfg["total_params_b"],
            "active_params_b": model_cfg["active_params_b"],
            "dataset_name": args.dataset,
            "strategy": args.strategy,
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
        rows.append(row)

        print(
            f"[{i}/{len(records)}] sample_id={rec['sample_id']} "
            f"correct={correct} latency={latency_sec:.3f}s tokens={n_output_tokens} tps={tps:.2f}"
        )

    out_path = RAW_DIR / f"single__{args.model}__{args.dataset}__{args.strategy}__n{len(records)}.csv"
    write_rows_csv(out_path, rows)
    unload_model(model)

    num_correct = sum(int(r["correct"]) for r in rows)
    n = len(rows)
    acc = (num_correct / n) if n else 0.0
    print(f"Saved: {out_path}")
    print(f"Accuracy: {num_correct}/{n} = {acc:.4f}")


if __name__ == "__main__":
    main()
