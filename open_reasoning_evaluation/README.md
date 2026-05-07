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
