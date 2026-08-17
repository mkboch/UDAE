# Open Reasoning Evaluation

This repository contains the core code, configuration files, prompt definitions, evaluation utilities, and curated result tables for the open reasoning model evaluation project.

## Structure

- `configs/`
  - Benchmark configuration files:
    - `datasets.yaml`
    - `models.yaml`
    - `prompts.yaml`

- `prompts/`
  - Prompt construction and few-shot demonstration files:
    - `builder.py`
    - `few_shot_examples.json`

- `evaluation/`
  - Output extraction, grading, and evaluation utilities:
    - `extractor.py`
    - `grader.py`
    - `metrics.py`

- `code/experiments/`
  - Benchmark execution scripts.

- `code/models/`
  - Model loading and inference support code.

- `code/utils/`
  - Utility scripts used by the benchmarking pipeline.

- `code/additional_analysis_scripts/`
  - Scripts used for additional robustness, audit, and prompt-protocol analyses.

- `results/unified_238/`
  - Final unified-size evaluation summaries for the 238-example matched protocol.

- `results/full500_revision/`
  - Revision-stage expanded evaluation summaries and audits.

- `results/full500_journal/`
  - Journal-oriented expanded evaluation summaries.

- `results/protocolfix_journal/`
  - Protocol-fix rerun summaries for prompt and token-budget analysis.

## Additional Updated Analyses

Additional analysis outputs are included under:

- `results/full_run_audit_v1/`
  - Expanded evaluation coverage audits for ARC-Challenge, GSM8K, TruthfulQA MC1, and MATH L1-L3.

- `results/experiments_1_to_5_check_v1/`
  - Robustness, parser/extraction, prompt-protocol, and statistical sensitivity analysis tables.

- `results/mc_prompt_ablation_v3_full500/`
  - Full-500 multiple-choice prompt-protocol ablation results comparing answer-only prompts with rationale-allowed final-answer prompts.

## Notes

This repository intentionally excludes manuscript-only assets such as paper figures, paper tables, and submission zip bundles. It serves as the public code-and-results companion for the paper.
