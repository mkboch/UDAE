# Unified Deployment-Aware Evaluation of Open Reasoning Language Models

This repository contains the public code, configuration files, selected benchmark records and identifiers, evaluation logic, software-environment records, and curated results associated with the paper **"Unified Deployment-Aware Evaluation of Open Reasoning Language Models."**

## Repository Structure

- `configs/`
  - `datasets.yaml`: 238-example matched-core configuration.
  - `datasets_larger_sample.yaml`: manifest for the reported larger-sample matrix: ARC-Challenge 500, GSM8K 500, MATH L1--L3 238, and TruthfulQA MC1 500.
  - `datasets_truthfulqa_followup.yaml`: retained dataset configuration from the final TruthfulQA follow-up stage.
  - `models.yaml`: model configuration.
  - `prompts.yaml`: prompt configuration.

- `prompts/`
  - `builder.py`: shared prompt construction.
  - `few_shot_examples.json`: retained few-shot demonstrations used by the final pipeline.

- `evaluation/`
  - `extractor.py`: strict task-specific answer extraction.
  - `grader.py`: grading logic.
  - `metrics.py`: evaluation utilities.

- `experiments/`
  - Benchmark execution scripts, including `run_benchmark_with_fallback.py`.

- `models/`
  - Model loading and inference support code.

- `data/`
  - `prepared/`: exact matched-core prepared records.
  - `indices/`: selected matched-core sample identifiers.
  - `larger_sample/prepared/`: prepared records for the reported larger-sample matrix.
  - `larger_sample/indices/`: selected identifiers for the larger-sample matrix.
  - `prepare_datasets.py`: retained general preparation script.
  - `README.md`: dataset provenance and reconstruction notes.

- `code/additional_analysis_scripts/`
  - Additional audit, robustness, and prompt-protocol analysis scripts.

- `results/unified_238/`
  - Matched 238-example evaluation summaries.

- `results/full500_revision/`
  - Revision-stage expanded evaluation summaries and audits.

- `results/full500_journal/`
  - Final expanded evaluation summaries.

- `results/protocolfix_journal/`
  - Prompt/token-budget protocol-fix summaries.

- `results/full_run_audit_v1/`
  - Expanded-run coverage and consistency audits.

- `results/experiments_1_to_5_check_v1/`
  - Robustness, parser/extraction, prompt-protocol, and statistical sensitivity analyses.

- `results/mc_prompt_ablation_v3_full500/`
  - Full-500 multiple-choice prompt-protocol ablation outputs and summaries.

- `environment/`
  - Retained software-environment and package-version records for the main-result and follow-up audit environments.

- `provenance/`
  - Retained audit/provenance material supporting the released artifact.

- `requirements.txt`
  - Python package requirements for the benchmark code.

## Reproducibility Notes

The matched-core evaluation uses 238 examples per benchmark. MATH L1--L3 contains exactly 238 Level 1--3 test examples, so the larger-sample robustness matrix retains all 238 MATH examples while ARC-Challenge, GSM8K, and TruthfulQA MC1 use 500 examples.

The final reported TruthfulQA results use corrected counterbalanced choice records, balanced few-shot demonstrations, and an A--Z multiple-choice extractor. The exact corrected TruthfulQA prepared records and selected identifiers are included under `data/`.

The retained main-result and follow-up audit runs were produced on the same NVIDIA H100 server under separate software environments. Corresponding environment records are included under `environment/`.

This repository intentionally excludes manuscript-only assets such as submission PDFs, paper figures, paper tables, and submission zip bundles.
