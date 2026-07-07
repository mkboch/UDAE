# MC Prompt Ablation V3 Full-500 Status

## Validation

| check | status |
| :--- | :--- |
| All 16 new full-500 rationale conditions present | PASS |
| All 24 copied full-500 baseline conditions present | PASS |
| No model inference error rows | PASS |
| New prompt load mode is bf16 direct/no 4-bit | PASS |
| V3 full-500, not 238 | PASS |

## Reviewer-facing prompt-ablation table

| model | dataset | comparison | old_strict_acc | new_strict_finalline_acc | delta_new_minus_old_strict | old_lenient_acc | new_lenient_acc | delta_new_minus_old_lenient | old_missing_prediction_rate | new_missing_prediction_rate | new_final_hash_answer_rate | old_bare_letter_response_rate | new_bare_letter_response_rate | old_think_tag_rate | new_think_tag_rate | old_mean_latency_sec | new_mean_latency_sec |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| gemma_4_26b_a4b | arc_challenge | orig_cot_answer_only -> cot_rationale_final_letter | 0.858000 | 0.956000 | 0.098000 | 0.804000 | 0.956000 | 0.152000 | 0.104000 | 0.014000 | 0.986000 | 0.512000 | 0.000000 | 0.000000 | 0.000000 | 2.510284 | 13.872989 |
| gemma_4_26b_a4b | arc_challenge | orig_few_shot_cot_answer_only -> few_shot_cot_rationale_final_letter | 0.912000 | 0.854000 | -0.058000 | 0.936000 | 0.878000 | -0.058000 | 0.000000 | 0.112000 | 0.888000 | 0.016000 | 0.000000 | 0.000000 | 0.000000 | 2.572086 | 7.868493 |
| gemma_4_26b_a4b | truthfulqa_mc1 | orig_cot_answer_only -> cot_rationale_final_letter | 0.750000 | 0.882000 | 0.132000 | 0.742000 | 0.886000 | 0.144000 | 0.132000 | 0.006000 | 0.994000 | 0.196000 | 0.000000 | 0.000000 | 0.000000 | 2.504601 | 13.600370 |
| gemma_4_26b_a4b | truthfulqa_mc1 | orig_few_shot_cot_answer_only -> few_shot_cot_rationale_final_letter | 0.820000 | 0.876000 | 0.056000 | 0.810000 | 0.882000 | 0.072000 | 0.034000 | 0.010000 | 0.990000 | 0.672000 | 0.000000 | 0.000000 | 0.000000 | 1.731547 | 6.500817 |
| gemma_4_e4b | arc_challenge | orig_cot_answer_only -> cot_rationale_final_letter | 0.878000 | 0.932000 | 0.054000 | 0.886000 | 0.936000 | 0.050000 | 0.018000 | 0.012000 | 0.988000 | 0.944000 | 0.000000 | 0.000000 | 0.000000 | 0.647962 | 16.683769 |
| gemma_4_e4b | arc_challenge | orig_few_shot_cot_answer_only -> few_shot_cot_rationale_final_letter | 0.884000 | 0.784000 | -0.100000 | 0.836000 | 0.802000 | -0.034000 | 0.002000 | 0.054000 | 0.946000 | 0.924000 | 0.000000 | 0.000000 | 0.000000 | 0.711367 | 6.934927 |
| gemma_4_e4b | truthfulqa_mc1 | orig_cot_answer_only -> cot_rationale_final_letter | 0.628000 | 0.730000 | 0.102000 | 0.590000 | 0.768000 | 0.178000 | 0.206000 | 0.056000 | 0.944000 | 0.714000 | 0.000000 | 0.000000 | 0.000000 | 2.826280 | 18.644742 |
| gemma_4_e4b | truthfulqa_mc1 | orig_few_shot_cot_answer_only -> few_shot_cot_rationale_final_letter | 0.742000 | 0.786000 | 0.044000 | 0.740000 | 0.814000 | 0.074000 | 0.024000 | 0.044000 | 0.956000 | 0.974000 | 0.000000 | 0.000000 | 0.000000 | 0.306933 | 6.720059 |
| qwen3_8b | arc_challenge | orig_cot_answer_only -> cot_rationale_final_letter | 0.312000 | 0.818000 | 0.506000 | 0.404000 | 0.886000 | 0.482000 | 0.008000 | 0.158000 | 0.842000 | 0.096000 | 0.000000 | 0.904000 | 0.130000 | 5.159804 | 7.228185 |
| qwen3_8b | arc_challenge | orig_few_shot_cot_answer_only -> few_shot_cot_rationale_final_letter | 0.508000 | 0.856000 | 0.348000 | 0.586000 | 0.900000 | 0.314000 | 0.020000 | 0.116000 | 0.884000 | 0.066000 | 0.000000 | 0.634000 | 0.098000 | 4.877230 | 6.358009 |
| qwen3_8b | truthfulqa_mc1 | orig_cot_answer_only -> cot_rationale_final_letter | 0.990000 | 0.506000 | -0.484000 | 0.440000 | 0.716000 | 0.276000 | 0.002000 | 0.454000 | 0.546000 | 0.044000 | 0.000000 | 0.954000 | 0.386000 | 5.192243 | 8.859072 |
| qwen3_8b | truthfulqa_mc1 | orig_few_shot_cot_answer_only -> few_shot_cot_rationale_final_letter | 0.986000 | 0.562000 | -0.424000 | 0.454000 | 0.710000 | 0.256000 | 0.002000 | 0.398000 | 0.602000 | 0.104000 | 0.000000 | 0.894000 | 0.330000 | 5.152842 | 8.502308 |
| phi_4_reasoning | arc_challenge | orig_cot_answer_only -> cot_rationale_final_letter | 0.268000 | 0.864000 | 0.596000 | 0.822000 | 0.902000 | 0.080000 | 0.000000 | 0.110000 | 0.890000 | 0.018000 | 0.000000 | 0.982000 | 0.976000 | 4.858918 | 10.573819 |
| phi_4_reasoning | arc_challenge | orig_few_shot_cot_answer_only -> few_shot_cot_rationale_final_letter | 0.246000 | 0.796000 | 0.550000 | 0.424000 | 0.852000 | 0.428000 | 0.000000 | 0.112000 | 0.888000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 4.784699 | 10.660771 |
| phi_4_reasoning | truthfulqa_mc1 | orig_cot_answer_only -> cot_rationale_final_letter | 0.994000 | 0.526000 | -0.468000 | 0.712000 | 0.802000 | 0.090000 | 0.002000 | 0.436000 | 0.564000 | 0.052000 | 0.000000 | 0.946000 | 0.986000 | 4.910404 | 10.580762 |
| phi_4_reasoning | truthfulqa_mc1 | orig_few_shot_cot_answer_only -> few_shot_cot_rationale_final_letter | 1.000000 | 0.424000 | -0.576000 | 0.500000 | 0.754000 | 0.254000 | 0.000000 | 0.560000 | 0.440000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 4.800454 | 10.656946 |

## Grouped summary

| dataset | strategy | mean_strict_acc | mean_diagnostic_lenient_acc | mean_missing_prediction_rate | mean_final_hash_answer_rate | mean_bare_letter_response_rate | mean_think_tag_rate | mean_latency_sec | total_error_rows |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| arc_challenge | cot_rationale_final_letter | 0.892500 | 0.920000 | 0.073500 | 0.926500 | 0.000000 | 0.276500 | 12.089690 | 0 |
| arc_challenge | few_shot_cot_rationale_final_letter | 0.822500 | 0.858000 | 0.098500 | 0.901500 | 0.000000 | 0.274500 | 7.955550 | 0 |
| arc_challenge | orig_cot_answer_only | 0.579000 | 0.729000 | 0.032500 | 0.000000 | 0.392500 | 0.471500 | 3.294242 | 0 |
| arc_challenge | orig_few_shot_cot_answer_only | 0.637500 | 0.695500 | 0.005500 | 0.355500 | 0.251500 | 0.408500 | 3.236346 | 0 |
| arc_challenge | orig_zero_shot_answer_only | 0.626500 | 0.788000 | 0.015000 | 0.000000 | 0.381000 | 0.425500 | 3.297762 | 0 |
| truthfulqa_mc1 | cot_rationale_final_letter | 0.661000 | 0.793000 | 0.238000 | 0.762000 | 0.000000 | 0.343000 | 12.921236 | 0 |
| truthfulqa_mc1 | few_shot_cot_rationale_final_letter | 0.662000 | 0.790000 | 0.253000 | 0.747000 | 0.000000 | 0.332500 | 8.095032 | 0 |
| truthfulqa_mc1 | orig_cot_answer_only | 0.840500 | 0.621000 | 0.085500 | 0.000000 | 0.251500 | 0.475000 | 3.858382 | 0 |
| truthfulqa_mc1 | orig_few_shot_cot_answer_only | 0.887000 | 0.626000 | 0.015000 | 0.004000 | 0.437500 | 0.473500 | 2.997944 | 0 |
| truthfulqa_mc1 | orig_zero_shot_answer_only | 0.870500 | 0.687000 | 0.036500 | 0.000000 | 0.352500 | 0.440500 | 3.155499 | 0 |

## Interpretation

The V3 ablation uses the full 500 ARC-Challenge and TruthfulQA MC1 examples from journal_full500_v1, not the earlier 238-example matched subset. It compares the original answer-only multiple-choice protocol against rationale-allowed prompts requiring a final `#### <letter>` answer line. All new inference runs use `bf16_direct_no_4bit`, avoiding quantization ambiguity.

## Files
- `review_response_runs/mc_prompt_ablation_v3_full500/posthoc_summary/v3_full500_condition_summary.csv`
- `review_response_runs/mc_prompt_ablation_v3_full500/posthoc_summary/v3_full500_grouped_summary.csv`
- `review_response_runs/mc_prompt_ablation_v3_full500/posthoc_summary/v3_full500_pairwise_prompt_comparison.csv`
- `review_response_runs/mc_prompt_ablation_v3_full500/posthoc_summary/v3_full500_reviewer_prompt_ablation_table.csv`
- `review_response_runs/mc_prompt_ablation_v3_full500/posthoc_summary/v3_full500_validation_checklist.csv`
- `review_response_runs/mc_prompt_ablation_v3_full500/posthoc_summary/v3_full500_condition_summary.md`
- `review_response_runs/mc_prompt_ablation_v3_full500/posthoc_summary/v3_full500_grouped_summary.md`
- `review_response_runs/mc_prompt_ablation_v3_full500/posthoc_summary/v3_full500_pairwise_prompt_comparison.md`
- `review_response_runs/mc_prompt_ablation_v3_full500/posthoc_summary/v3_full500_reviewer_prompt_ablation_table.md`
- `review_response_runs/mc_prompt_ablation_v3_full500/posthoc_summary/v3_full500_validation_checklist.md`