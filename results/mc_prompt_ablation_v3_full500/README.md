# MC Prompt Ablation V3 Full-500

This folder contains the full-500 multiple-choice prompt-protocol ablation.

Purpose:
- Compare answer-only multiple-choice prompting with rationale-allowed prompting.
- The rationale-allowed prompts permit short reasoning but require a final `#### <letter>` answer line.
- The ablation uses the 500-example ARC-Challenge and TruthfulQA MC1 item sets from `journal_full500_v1`.
- New inference was run with `bf16_direct_no_4bit`.

Main files:
- `MC_PROMPT_ABLATION_V3_FULL500_STATUS.md`
- `v3_full500_prompt_ablation_table.csv`
- `v3_full500_grouped_summary.csv`
- `v3_full500_condition_summary.csv`
- `v3_full500_validation_checklist.csv`
- `raw_outputs/` contains compact raw CSV outputs when file sizes are safe for GitHub.
