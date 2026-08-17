import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "datasets.yaml"
INDICES_DIR = ROOT / "data" / "indices"
PROMPTS_PATH = ROOT / "prompts" / "few_shot_examples.json"


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def choose_indices(n_total: int, sample_size: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    sample_size = min(sample_size, n_total)
    indices = list(range(n_total))
    rng.shuffle(indices)
    chosen = sorted(indices[:sample_size])
    return chosen


def format_arc_question(example: Dict[str, Any]) -> str:
    q = example["question"].strip()
    choices = example["choices"]
    labels = choices["label"]
    texts = choices["text"]
    lines = [q]
    for lab, txt in zip(labels, texts):
        lines.append(f"{lab}. {txt}")
    return "\n".join(lines)


def format_truthfulqa_question(example: Dict[str, Any]) -> str:
    q = example["question"].strip()
    targets = example["mc1_targets"]
    choices = targets["choices"]
    lines = [q]
    for i, choice in enumerate(choices):
        label = chr(ord("A") + i)
        lines.append(f"{label}. {choice}")
    lines.append("Choose the most truthful answer.")
    return "\n".join(lines)


def get_gold_truthfulqa_label(example: Dict[str, Any]) -> Optional[str]:
    labels = example["mc1_targets"]["labels"]
    for i, value in enumerate(labels):
        if value == 1:
            return chr(ord("A") + i)
    return None


def maybe_filter_math_levels(dataset, allowed_levels=("Level 1", "Level 2", "Level 3", "1", "2", "3", 1, 2, 3)):
    if "level" not in dataset.column_names:
        print("MATH dataset has no 'level' column. Using full test split.")
        return dataset

    try:
        unique_levels = sorted({str(x["level"]) for x in dataset})
        print(f"MATH detected levels: {unique_levels}")
    except Exception as e:
        print(f"Could not enumerate MATH levels cleanly: {e}")

    filtered = dataset.filter(lambda x: str(x["level"]) in {str(v) for v in allowed_levels})

    if len(filtered) == 0:
        print("Filtered MATH subset is empty. Falling back to full test split.")
        return dataset

    print(f"MATH filtered subset size: {len(filtered)} / {len(dataset)}")
    return filtered


def normalize_record(dataset_name: str, example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    if dataset_name == "gsm8k":
        return {
            "sample_id": idx,
            "question": example["question"],
            "gold_answer": example["answer"],
            "metadata": {},
        }

    if dataset_name == "math_l1_l3":
        metadata = {}
        if "level" in example:
            metadata["level"] = example["level"]
        if "type" in example:
            metadata["type"] = example["type"]
        return {
            "sample_id": idx,
            "question": example["problem"],
            "gold_answer": example["solution"],
            "metadata": metadata,
        }

    if dataset_name == "arc_challenge":
        return {
            "sample_id": idx,
            "question": format_arc_question(example),
            "gold_answer": example["answerKey"],
            "metadata": {},
        }

    if dataset_name == "truthfulqa_mc1":
        return {
            "sample_id": idx,
            "question": format_truthfulqa_question(example),
            "gold_answer": get_gold_truthfulqa_label(example),
            "metadata": {},
        }

    raise ValueError(f"Unsupported dataset name: {dataset_name}")


def save_indices_csv(dataset_name: str, selected_indices: List[int]) -> Path:
    ensure_dir(INDICES_DIR)
    out_path = INDICES_DIR / f"{dataset_name}_indices.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id"])
        for idx in selected_indices:
            writer.writerow([idx])
    return out_path


def save_prepared_jsonl(dataset_name: str, records: List[Dict[str, Any]]) -> Path:
    out_dir = ROOT / "data" / "prepared"
    ensure_dir(out_dir)
    out_path = out_dir / f"{dataset_name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out_path


def main() -> None:
    cfg = load_yaml(CONFIG_PATH)
    datasets_cfg = cfg["datasets"]

    for dcfg in datasets_cfg:
        print(f"\nPreparing dataset: {dcfg['name']}")
        ds = load_dataset(
            dcfg["hf_path"],
            dcfg["hf_name"],
            split=dcfg["split"],
        )

        if dcfg["name"] == "math_l1_l3":
            ds = maybe_filter_math_levels(ds)

        selected_indices = choose_indices(
            n_total=len(ds),
            sample_size=int(dcfg["sample_size"]),
            seed=int(dcfg["seed"]),
        )

        records = []
        for idx in selected_indices:
            ex = ds[int(idx)]
            records.append(normalize_record(dcfg["name"], ex, int(idx)))

        idx_path = save_indices_csv(dcfg["name"], selected_indices)
        data_path = save_prepared_jsonl(dcfg["name"], records)

        print(f"Saved indices: {idx_path}")
        print(f"Saved prepared data: {data_path}")
        print(f"Total records: {len(records)}")


if __name__ == "__main__":
    main()
