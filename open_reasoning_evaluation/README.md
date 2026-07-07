# Open Reasoning Evaluation

This repository folder contains the core code, configuration files, and curated result tables for the open reasoning model evaluation project.

## Structure

- `configs/`
  - Benchmark configuration files:
    - `datasets.yaml`
    - `models.yaml`
    - `prompts.yaml`

- `code/experiments/`
  - Benchmark execution scripts.

- `code/models/`
  - Model loading and inference support code.

- `code/utils/`
  - Utility scripts used by the benchmarking pipeline.

- `results/unified_238/`
  - Final unified-size evaluation summaries for the 238-example matched protocol.

- `results/full500_revision/`
  - Revision-stage 500-sample summaries and audits.

- `results/full500_journal/`
  - Journal-oriented expanded summaries.

- `results/protocolfix_journal/`
  - Protocol-fix rerun summaries for prompt/token-budget analysis.

## Notes

This upload intentionally excludes manuscript-only assets such as paper figures, paper tables, and zip bundles. It is meant to serve as a clean code-and-results companion for the paper.

## Revision-stage evidence added after review

Additional reviewer-response evidence is included under:

- `results/full_run_audit_v1/`
  - verifies the expanded evaluation coverage: ARC-Challenge, GSM8K, and TruthfulQA MC1 at 500 examples, with MATH L1-L3 at 238 examples.
- `results/experiments_1_to_5_check_v1/`
  - checklist and tables for sample-size robustness, parser diagnostics, prompt-protocol analysis, and statistical testing.
- `results/mc_prompt_ablation_v3_full500/`
  - full-500 multiple-choice prompt-protocol ablation comparing answer-only prompts against rationale-allowed final-answer prompts.
- `results/final_revision_evidence_v1/`
  - consolidated evidence package for the revised manuscript and response letter.
