# MC Prompt Ablation V3 Full-500

This folder contains the revision-stage full-500 multiple-choice prompt-protocol ablation.

Purpose:
- Addresses reviewer concern that the original multiple-choice CoT prompts mixed "think step by step" with "return only one capital letter."
- Compares original answer-only multiple-choice prompts with rationale-allowed prompts requiring a final `#### <letter>` answer line.
- Uses full 500-example ARC-Challenge and TruthfulQA MC1 item sets from `journal_full500_v1`, not the earlier 238-example matched subset.
- New inference was run with `bf16_direct_no_4bit`.

Main files:
- `MC_PROMPT_ABLATION_V3_FULL500_STATUS.md`
- `v3_full500_reviewer_prompt_ablation_table.csv`
- `v3_full500_grouped_summary.csv`
- `v3_full500_condition_summary.csv`
- `v3_full500_validation_checklist.csv`
- `raw_outputs/` contains compact raw CSV outputs when file sizes are safe for GitHub.
