# LLM Pareto Workshop Package

This folder contains the main artifacts selected for upload from the `llm-pareto-2026` project.

## Included contents

- `results/workshop_unified238_v1_final/`
  - Final unified 238-sample benchmark outputs used for the workshop-style paper version.
  - Includes:
    - aggregated CSV
    - weighted summary CSV
    - prompting accuracy CSV
    - prompting latency CSV
    - raw concatenated CSV
    - summary report

- `results/workshop_unified238_v1_post/`
  - Post-run logs and progress artifacts for the unified workshop evaluation.

- `results/revision_full500_v1_post/`
  - Full 500-sample comparison summaries used in earlier analysis.

- `results/journal_full500_v1_post/`
  - Journal-expansion run summaries.

- `results/journal_protocolfix_v1_post/`
  - Protocol-fix rerun summaries.

- `configs/`
  - `datasets.yaml`
  - `prompts.yaml`
  - `models.yaml`

- `paper_assets/`
  - packaged figures/tables zip for Overleaf upload

## Notes

- This package intentionally prioritizes summarized and paper-facing artifacts over the full raw run directory.
- The prompt templates reported in the manuscript appendix are grounded in `configs/prompts.yaml`.
- Generated on: 20260507_112122
