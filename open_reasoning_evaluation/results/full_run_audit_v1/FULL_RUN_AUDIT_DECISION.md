# Full Run Audit Decision

## journal_full500_v1 dataset coverage

| dataset        |   expected_conditions |   expected_n |   conditions_present |   min_n_from_filename |   max_n_from_filename |   min_n_rows |   max_n_rows | status   |
|:---------------|----------------------:|-------------:|---------------------:|----------------------:|----------------------:|-------------:|-------------:|:---------|
| arc_challenge  |                    21 |          500 |                   21 |                   500 |                   500 |          500 |          500 | PASS     |
| gsm8k          |                    21 |          500 |                   21 |                   500 |                   500 |          500 |          500 | PASS     |
| math_l1_l3     |                    21 |          238 |                   21 |                   238 |                   238 |          238 |          238 | PASS     |
| truthfulqa_mc1 |                    21 |          500 |                   21 |                   500 |                   500 |          500 |          500 | PASS     |

## Full-500 MC prompt-ablation baseline availability

| model           | dataset        | baseline_strategy   |   required_for_full500_prompt_ablation |   present |   n_rows | status   | file                                                                                    |
|:----------------|:---------------|:--------------------|---------------------------------------:|----------:|---------:|:---------|:----------------------------------------------------------------------------------------|
| gemma_4_26b_a4b | arc_challenge  | zero_shot           |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__gemma_4_26b_a4b__arc_challenge__zero_shot__n500.csv     |
| gemma_4_26b_a4b | arc_challenge  | cot                 |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__gemma_4_26b_a4b__arc_challenge__cot__n500.csv           |
| gemma_4_26b_a4b | arc_challenge  | few_shot_cot        |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__gemma_4_26b_a4b__arc_challenge__few_shot_cot__n500.csv  |
| gemma_4_26b_a4b | truthfulqa_mc1 | zero_shot           |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__gemma_4_26b_a4b__truthfulqa_mc1__zero_shot__n500.csv    |
| gemma_4_26b_a4b | truthfulqa_mc1 | cot                 |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__gemma_4_26b_a4b__truthfulqa_mc1__cot__n500.csv          |
| gemma_4_26b_a4b | truthfulqa_mc1 | few_shot_cot        |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__gemma_4_26b_a4b__truthfulqa_mc1__few_shot_cot__n500.csv |
| gemma_4_e4b     | arc_challenge  | zero_shot           |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__gemma_4_e4b__arc_challenge__zero_shot__n500.csv         |
| gemma_4_e4b     | arc_challenge  | cot                 |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__gemma_4_e4b__arc_challenge__cot__n500.csv               |
| gemma_4_e4b     | arc_challenge  | few_shot_cot        |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__gemma_4_e4b__arc_challenge__few_shot_cot__n500.csv      |
| gemma_4_e4b     | truthfulqa_mc1 | zero_shot           |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__gemma_4_e4b__truthfulqa_mc1__zero_shot__n500.csv        |
| gemma_4_e4b     | truthfulqa_mc1 | cot                 |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__gemma_4_e4b__truthfulqa_mc1__cot__n500.csv              |
| gemma_4_e4b     | truthfulqa_mc1 | few_shot_cot        |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__gemma_4_e4b__truthfulqa_mc1__few_shot_cot__n500.csv     |
| qwen3_8b        | arc_challenge  | zero_shot           |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__qwen3_8b__arc_challenge__zero_shot__n500.csv            |
| qwen3_8b        | arc_challenge  | cot                 |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__qwen3_8b__arc_challenge__cot__n500.csv                  |
| qwen3_8b        | arc_challenge  | few_shot_cot        |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__qwen3_8b__arc_challenge__few_shot_cot__n500.csv         |
| qwen3_8b        | truthfulqa_mc1 | zero_shot           |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__qwen3_8b__truthfulqa_mc1__zero_shot__n500.csv           |
| qwen3_8b        | truthfulqa_mc1 | cot                 |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__qwen3_8b__truthfulqa_mc1__cot__n500.csv                 |
| qwen3_8b        | truthfulqa_mc1 | few_shot_cot        |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__qwen3_8b__truthfulqa_mc1__few_shot_cot__n500.csv        |
| phi_4_reasoning | arc_challenge  | zero_shot           |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__phi_4_reasoning__arc_challenge__zero_shot__n500.csv     |
| phi_4_reasoning | arc_challenge  | cot                 |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__phi_4_reasoning__arc_challenge__cot__n500.csv           |
| phi_4_reasoning | arc_challenge  | few_shot_cot        |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__phi_4_reasoning__arc_challenge__few_shot_cot__n500.csv  |
| phi_4_reasoning | truthfulqa_mc1 | zero_shot           |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__phi_4_reasoning__truthfulqa_mc1__zero_shot__n500.csv    |
| phi_4_reasoning | truthfulqa_mc1 | cot                 |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__phi_4_reasoning__truthfulqa_mc1__cot__n500.csv          |
| phi_4_reasoning | truthfulqa_mc1 | few_shot_cot        |                                      1 |         1 |      500 | PASS     | results/raw/journal_full500_v1__phi_4_reasoning__truthfulqa_mc1__few_shot_cot__n500.csv |

## Decision

PASS: The larger/full run exists for ARC-Challenge, GSM8K, and TruthfulQA MC1 at n=500, with MATH L1-L3 retained at n=238. Full-500 MC prompt ablation can be run for ARC-Challenge and TruthfulQA MC1.

- all_full_ok: True
- mc_baselines_ok: True