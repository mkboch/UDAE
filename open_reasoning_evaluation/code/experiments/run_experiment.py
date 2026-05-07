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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_prepared_records(dataset_name: str) -> List[Dict[str, Any]]:
    path = ROOT / "data" / "prepared" / f"{dataset_name}.jsonl"
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def filter_models(all_models: List[Dict[str, Any]], wanted: List[str]) -> List[Dict[str, Any]]:
    if not wanted:
        return all_models
    wanted_set = set(wanted)
    selected = [m for m in all_models if m["name"] in wanted_set]
    missing = wanted_set - {m["name"] for m in selected}
    if missing:
        raise ValueError(f"Unknown model names: {sorted(missing)}")
    return selected


def filter_datasets(all_datasets: List[Dict[str, Any]], wanted: List[str]) -> List[Dict[str, Any]]:
    if not wanted:
        return all_datasets
    wanted_set = set(wanted)
    selected = [d for d in all_datasets if d["name"] in wanted_set]
    missing = wanted_set - {d["name"] for d in selected}
    if missing:
        raise ValueError(f"Unknown dataset names: {sorted(missing)}")
    return selected


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
    parser.add_argument("--models", nargs="*", default=[], help="Subset of model config names")
    parser.add_argument("--datasets", nargs="*", default=[], help="Subset of dataset config names")
    parser.add_argument("--strategies", nargs="*", default=["zero_shot"], choices=["zero_shot", "cot", "few_shot_cot"])
    parser.add_argument("--limit", type=int, default=0, help="Optional cap per dataset, 0 means all")
    parser.add_argument("--tag", type=str, default="manual")
    parser.add_argument("--max-new-tokens", type=int, default=0)
    args = parser.parse_args()

    models = filter_models(load_yaml(MODELS_CFG)["models"], args.models)
    datasets = filter_datasets(load_yaml(DATASETS_CFG)["datasets"], args.datasets)
    gen_cfg = get_generation_config()

    all_rows: List[Dict[str, Any]] = []

    for model_cfg in models:
        print(f"\n=== Loading model: {model_cfg['name']} ({model_cfg['hf_id']}) ===")
        model, tokenizer, peak_vram_gb = load_model(model_cfg)
        print(f"Peak VRAM after load: {peak_vram_gb:.3f} GB")

        try:
            for dataset_cfg in datasets:
                dataset_name = dataset_cfg["name"]
                records = load_prepared_records(dataset_name)
                if int(args.limit) > 0:
                    records = records[: int(args.limit)]

                for strategy in args.strategies:
                    print(
                        f"\n--- model={model_cfg['name']} dataset={dataset_name} "
                        f"strategy={strategy} n={len(records)} ---"
                    )

                    run_rows: List[Dict[str, Any]] = []
                    for i, rec in enumerate(records, start=1):
                        prompt = build_prompt(strategy, rec["question"], dataset_name)
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
                        correct = grade_prediction(prediction, rec["gold_answer"], dataset_name)
                        tps = tokens_per_second(n_output_tokens, latency_sec)

                        row = {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "model_name": model_cfg["name"],
                            "model_pretty_name": model_cfg["pretty_name"],
                            "hf_id": model_cfg["hf_id"],
                            "architecture": model_cfg["architecture"],
                            "total_params_b": model_cfg["total_params_b"],
                            "active_params_b": model_cfg["active_params_b"],
                            "dataset_name": dataset_name,
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

                        print(
                            f"[{i}/{len(records)}] sample_id={rec['sample_id']} "
                            f"correct={correct} latency={latency_sec:.3f}s tokens={n_output_tokens} tps={tps:.2f}"
                        )

                    out_path = (
                        RAW_DIR
                        / f"{args.tag}__{model_cfg['name']}__{dataset_name}__{strategy}__n{len(run_rows)}.csv"
                    )
                    write_rows_csv(out_path, run_rows)

                    num_correct = sum(int(r["correct"]) for r in run_rows)
                    n = len(run_rows)
                    acc = (num_correct / n) if n else 0.0
                    print(f"Saved: {out_path}")
                    print(f"Accuracy: {num_correct}/{n} = {acc:.4f}")

        finally:
            unload_model(model)

    summary_path = RAW_DIR / f"{args.tag}__ALL_RUNS__n{len(all_rows)}.csv"
    write_rows_csv(summary_path, all_rows)
    print(f"\nSaved combined raw file: {summary_path}")


if __name__ == "__main__":
    main()
