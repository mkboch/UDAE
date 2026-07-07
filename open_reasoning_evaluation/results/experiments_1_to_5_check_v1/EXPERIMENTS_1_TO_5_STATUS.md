# Experiments 1–5 Check Summary

This check excludes experiment 6, the reproducibility package audit, which can be completed later before final revision/upload.

## Checklist

| experiment                                               | status                                             | evidence_file                                                                             | paper_action                                                                                                       |
|:---------------------------------------------------------|:---------------------------------------------------|:------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|
| 1 larger/full-size robustness beyond submitted n=238     | PASS                                               | exp1_larger_size_coverage_by_dataset.md; exp1_original_vs_extended_weighted_comparison.md | Add larger non-MATH robustness analysis; explain MATH L1-L3 matched cap.                                           |
| 2 repeated-subset / bootstrap rank stability             | PASS_EXISTING_TABLE_FOUND                          | exp2_rank_stability_original_and_extended.md                                              | Add winner frequency, interval, variance/stability table.                                                          |
| 3 Phi parser-aware diagnostic                            | PASS                                               | exp3_phi4_parser_artifact_diagnostics.md                                                  | Separate parser failures from wrong answers; revise Phi claim.                                                     |
| 4 MC prompt protocol diagnostic                          | NO_NEW_GPU_ABLATION_FOUND_TEXTUAL_REFRAMING_NEEDED | exp4_mc_prompt_protocol_response_diagnostics.md; exp4_math_only_prompt_rank_range.md      | Clarify MC CoT was answer-format-constrained; weaken prompt-sensitivity claims or run optional GPU ablation later. |
| 5 bootstrap/permutation procedure + multiple comparisons | PASS                                               | exp5_holm_corrected_paired_permutation_tests.md                                           | Explain paired sign-flip permutation, resampling unit, and Holm correction.                                        |

## Key outputs
- `review_response_runs/experiments_1_to_5_check_v1/EXPERIMENTS_1_TO_5_CHECKLIST.md`
- `review_response_runs/experiments_1_to_5_check_v1/exp1_extended_inventory.md`
- `review_response_runs/experiments_1_to_5_check_v1/exp1_extended_weighted_scores.md`
- `review_response_runs/experiments_1_to_5_check_v1/exp1_larger_size_coverage_by_dataset.md`
- `review_response_runs/experiments_1_to_5_check_v1/exp1_original_n238_inventory.md`
- `review_response_runs/experiments_1_to_5_check_v1/exp1_original_vs_extended_weighted_comparison.md`
- `review_response_runs/experiments_1_to_5_check_v1/exp1_original_weighted_scores.md`
- `review_response_runs/experiments_1_to_5_check_v1/exp2_rank_stability_original_and_extended.md`
- `review_response_runs/experiments_1_to_5_check_v1/exp3_phi4_parser_artifact_diagnostics.md`
- `review_response_runs/experiments_1_to_5_check_v1/exp3_top_parser_artifact_rates_all_conditions.md`
- `review_response_runs/experiments_1_to_5_check_v1/exp4_math_only_prompt_rank_range.md`
- `review_response_runs/experiments_1_to_5_check_v1/exp4_math_only_prompt_sensitivity_by_strategy.md`
- `review_response_runs/experiments_1_to_5_check_v1/exp4_mc_prompt_protocol_response_diagnostics.md`
- `review_response_runs/experiments_1_to_5_check_v1/exp5_holm_corrected_paired_permutation_tests.md`

## Main interpretation

- Larger-size robustness: use the extended non-MATH n=500 results and repeated 238-subset analysis to address the one-seed/n=238 concern.
- Precision: already handled by prior precision table; all submitted conditions record bf16.
- Phi: revise from clean reasoning failure to parser/interface-compatibility diagnostic.
- Prompt protocol: current evidence supports a textual reframing; no new GPU prompt ablation was found in existing results.
- Statistics: use Holm-adjusted paired permutation table and clearly state the paired resampling/sign-flip design.