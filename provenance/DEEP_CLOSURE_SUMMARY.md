# Deep Closure No-GPU Evidence Summary

## Checklist

| check | status | detail |
| --- | --- | --- |
| larger_selected_conditions | PASS | 84/84 |
| larger_arc_challenge_coverage | PASS | conditions=21, n_values=[500] |
| larger_gsm8k_coverage | PASS | conditions=21, n_values=[500] |
| larger_math_l1_l3_coverage | PASS | conditions=21, n_values=[238] |
| larger_truthfulqa_mc1_coverage | PASS | conditions=21, n_values=[500] |
| repeated_subset_iterations | PASS | 1000/1000 |
| weight_sensitivity_top3 | PASS | rows=36 |
| precision_bf16_footprint | PASS | rows=7 |
| hf_snapshot_manifest | PASS | rows=7 |

## Repeated-subset stability: top 10

| model | model_pretty | strategy | mean_weighted_score | std_weighted_score | ci_low_2p5 | ci_high_97p5 | winner_frequency | budget16_winner_frequency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma_4_26b_a4b | Gemma-4-26B-A4B | zero_shot | 0.7834 | 0.0082 | 0.7681 | 0.7996 | 1.0000 | 0.0000 |
| gemma_4_e4b | Gemma-4-E4B | few_shot_cot | 0.7537 | 0.0090 | 0.7370 | 0.7706 | 0.0000 | 0.4560 |
| gemma_4_e4b | Gemma-4-E4B | cot | 0.7524 | 0.0087 | 0.7353 | 0.7693 | 0.0000 | 0.2940 |
| gemma_4_e4b | Gemma-4-E4B | zero_shot | 0.7521 | 0.0084 | 0.7357 | 0.7685 | 0.0000 | 0.2500 |
| gemma_4_26b_a4b | Gemma-4-26B-A4B | cot | 0.7409 | 0.0090 | 0.7231 | 0.7584 | 0.0000 | 0.0000 |
| qwen3_8b | Qwen3-8B | few_shot_cot | 0.7206 | 0.0086 | 0.7034 | 0.7370 | 0.0000 | 0.0000 |
| gemma_4_26b_a4b | Gemma-4-26B-A4B | few_shot_cot | 0.7095 | 0.0093 | 0.6912 | 0.7277 | 0.0000 | 0.0000 |
| gemma_4_e2b | Gemma-4-E2B | few_shot_cot | 0.6912 | 0.0096 | 0.6727 | 0.7097 | 0.0000 | 0.0000 |
| qwen3_30b_a3b | Qwen3-30B-A3B | few_shot_cot | 0.6781 | 0.0092 | 0.6597 | 0.6966 | 0.0000 | 0.0000 |
| gemma_4_e2b | Gemma-4-E2B | zero_shot | 0.6615 | 0.0099 | 0.6416 | 0.6815 | 0.0000 | 0.0000 |

## Repeated-subset global summary

| n_iterations | subset_size_per_dataset | mean_spearman_vs_core_ranking | std_spearman_vs_core_ranking |
| --- | --- | --- | --- |
| 1000.0000 | 238.0000 | 0.9922 | 0.0036 |

## Larger-sample per-condition table: first 20 rows

| dataset | model_pretty | strategy | n | accuracy | wilson_low | wilson_high | mean_latency_sec | peak_vram_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arc_challenge | Gemma-4-26B-A4B | cot | 500 | 0.8580 | 0.8247 | 0.8859 | 2.5103 | 48.0674 |
| arc_challenge | Gemma-4-E2B | cot | 500 | 0.7660 | 0.7269 | 0.8010 | 0.4658 | 9.5431 |
| arc_challenge | Gemma-4-E4B | cot | 500 | 0.8780 | 0.8464 | 0.9038 | 0.6480 | 14.8945 |
| arc_challenge | Phi-4-mini-reasoning | cot | 500 | 0.2480 | 0.2122 | 0.2877 | 4.0515 | 7.1451 |
| arc_challenge | Phi-4-reasoning | cot | 500 | 0.2680 | 0.2311 | 0.3085 | 4.8589 | 27.3055 |
| arc_challenge | Qwen3-30B-A3B | cot | 500 | 0.3720 | 0.3308 | 0.4152 | 9.5387 | 57.6742 |
| arc_challenge | Qwen3-8B | cot | 500 | 0.3120 | 0.2730 | 0.3539 | 5.1598 | 15.2565 |
| arc_challenge | Gemma-4-26B-A4B | few_shot_cot | 500 | 0.9120 | 0.8839 | 0.9338 | 2.5721 | 48.0674 |
| arc_challenge | Gemma-4-E2B | few_shot_cot | 500 | 0.7300 | 0.6894 | 0.7671 | 4.1918 | 9.5431 |
| arc_challenge | Gemma-4-E4B | few_shot_cot | 500 | 0.8840 | 0.8530 | 0.9092 | 0.7114 | 14.8945 |
| arc_challenge | Phi-4-mini-reasoning | few_shot_cot | 500 | 0.2700 | 0.2329 | 0.3106 | 4.0634 | 7.1451 |
| arc_challenge | Phi-4-reasoning | few_shot_cot | 500 | 0.2460 | 0.2103 | 0.2856 | 4.7847 | 27.3055 |
| arc_challenge | Qwen3-30B-A3B | few_shot_cot | 500 | 0.4500 | 0.4069 | 0.4938 | 9.2271 | 57.6713 |
| arc_challenge | Qwen3-8B | few_shot_cot | 500 | 0.5080 | 0.4643 | 0.5516 | 4.8772 | 15.2565 |
| arc_challenge | Gemma-4-26B-A4B | zero_shot | 500 | 0.9220 | 0.8951 | 0.9424 | 3.4456 | 48.0674 |
| arc_challenge | Gemma-4-E2B | zero_shot | 500 | 0.7620 | 0.7228 | 0.7972 | 1.6288 | 9.5431 |
| arc_challenge | Gemma-4-E4B | zero_shot | 500 | 0.9060 | 0.8772 | 0.9286 | 0.1893 | 14.8945 |
| arc_challenge | Phi-4-mini-reasoning | zero_shot | 500 | 0.2480 | 0.2122 | 0.2877 | 4.0592 | 7.1451 |
| arc_challenge | Phi-4-reasoning | zero_shot | 500 | 0.2740 | 0.2367 | 0.3147 | 4.8329 | 27.3055 |
| arc_challenge | Qwen3-30B-A3B | zero_shot | 500 | 0.3840 | 0.3424 | 0.4274 | 9.4341 | 57.6733 |

## Weight sensitivity: top 3 per scheme/source

| source | scheme | rank | model_pretty | strategy | weighted_score |
| --- | --- | --- | --- | --- | --- |
| core | arc_heavy | 1 | Gemma-4-26B-A4B | zero_shot | 0.8622 |
| core | arc_heavy | 2 | Gemma-4-26B-A4B | few_shot_cot | 0.8210 |
| core | arc_heavy | 3 | Gemma-4-E4B | zero_shot | 0.8179 |
| core | equal | 1 | Gemma-4-26B-A4B | zero_shot | 0.8067 |
| core | equal | 2 | Gemma-4-E4B | few_shot_cot | 0.7689 |
| core | equal | 3 | Gemma-4-E4B | zero_shot | 0.7637 |
| core | gsm8k_heavy | 1 | Gemma-4-26B-A4B | zero_shot | 0.8017 |
| core | gsm8k_heavy | 2 | Gemma-4-E4B | zero_shot | 0.7742 |
| core | gsm8k_heavy | 3 | Qwen3-8B | few_shot_cot | 0.7721 |
| core | math_heavy | 1 | Gemma-4-26B-A4B | zero_shot | 0.7613 |
| core | math_heavy | 2 | Gemma-4-E4B | few_shot_cot | 0.7387 |
| core | math_heavy | 3 | Gemma-4-E4B | cot | 0.7160 |
| core | paper_original | 1 | Gemma-4-26B-A4B | zero_shot | 0.7941 |
| core | paper_original | 2 | Gemma-4-E4B | few_shot_cot | 0.7609 |
| core | paper_original | 3 | Gemma-4-E4B | cot | 0.7588 |
| core | truthfulqa_heavy | 1 | Qwen3-8B | few_shot_cot | 0.8359 |
| core | truthfulqa_heavy | 2 | Qwen3-30B-A3B | few_shot_cot | 0.8176 |
| core | truthfulqa_heavy | 3 | Qwen3-30B-A3B | zero_shot | 0.8080 |
| larger_sample | arc_heavy | 1 | Gemma-4-26B-A4B | zero_shot | 0.8469 |
| larger_sample | arc_heavy | 2 | Gemma-4-E4B | zero_shot | 0.8184 |
| larger_sample | arc_heavy | 3 | Gemma-4-26B-A4B | few_shot_cot | 0.8129 |
| larger_sample | equal | 1 | Gemma-4-26B-A4B | zero_shot | 0.7968 |
| larger_sample | equal | 2 | Gemma-4-E4B | few_shot_cot | 0.7643 |
| larger_sample | equal | 3 | Gemma-4-E4B | zero_shot | 0.7601 |
| larger_sample | gsm8k_heavy | 1 | Gemma-4-26B-A4B | zero_shot | 0.7901 |
| larger_sample | gsm8k_heavy | 2 | Qwen3-8B | few_shot_cot | 0.7684 |
| larger_sample | gsm8k_heavy | 3 | Gemma-4-E4B | zero_shot | 0.7648 |
| larger_sample | math_heavy | 1 | Gemma-4-26B-A4B | zero_shot | 0.7554 |
| larger_sample | math_heavy | 2 | Gemma-4-E4B | few_shot_cot | 0.7359 |
| larger_sample | math_heavy | 3 | Gemma-4-E4B | cot | 0.7109 |
| larger_sample | paper_original | 1 | Gemma-4-26B-A4B | zero_shot | 0.7836 |
| larger_sample | paper_original | 2 | Gemma-4-E4B | few_shot_cot | 0.7542 |
| larger_sample | paper_original | 3 | Gemma-4-E4B | cot | 0.7524 |
| larger_sample | truthfulqa_heavy | 1 | Qwen3-8B | few_shot_cot | 0.8380 |
| larger_sample | truthfulqa_heavy | 2 | Qwen3-30B-A3B | few_shot_cot | 0.8145 |
| larger_sample | truthfulqa_heavy | 3 | Qwen3-30B-A3B | zero_shot | 0.8050 |

## Precision/bf16 footprint audit

| model | model_pretty | total_params_b | active_params_b | recorded_load_modes | observed_peak_vram_gb | approx_bf16_param_footprint_gb | observed_vram_over_bf16_footprint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3_30b_a3b | Qwen3-30B-A3B | 30.0000 | 3.0000 | bf16 | 57.6205 | 60.0000 | 0.9603 |
| gemma_4_26b_a4b | Gemma-4-26B-A4B | 26.0000 | 3.8000 | bf16 | 48.0674 | 52.0000 | 0.9244 |
| phi_4_reasoning | Phi-4-reasoning | 14.0000 | 14.0000 | bf16 | 27.3055 | 28.0000 | 0.9752 |
| qwen3_8b | Qwen3-8B | 8.0000 | 8.0000 | bf16 | 15.2565 | 16.0000 | 0.9535 |
| gemma_4_e4b | Gemma-4-E4B | 8.0000 | 4.0000 | bf16 | 14.8945 | 16.0000 | 0.9309 |
| gemma_4_e2b | Gemma-4-E2B | 5.0000 | 2.0000 | bf16 | 9.5431 | 10.0000 | 0.9543 |
| phi_4_mini_reasoning | Phi-4-mini-reasoning | 3.8000 | 3.8000 | bf16 | 7.1451 | 7.6000 | 0.9402 |

## HF snapshot manifest

| hf_id | snapshot_hashes_found | snapshot_paths_found |
| --- | --- | --- |
| Qwen/Qwen3-30B-A3B |  |  |
| Qwen/Qwen3-8B |  |  |
| google/gemma-4-26B-A4B-it | 20da991ab4afab98e8f910c4a2e8f4fbefc404ad; 47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301; 7d4c97e54145f8ffd1a4dd1b4986a5015a517842 | /home/manikm/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/20da991ab4afab98e8f910c4a2e8f4fbefc404ad; /home/manikm/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301; /home/manikm/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/7d4c97e54145f8ffd1a4dd1b4986a5015a517842 |
| google/gemma-4-E2B-it |  |  |
| google/gemma-4-E4B-it |  |  |
| microsoft/Phi-4-mini-reasoning | 0e3b1e2d02ee478a3743abe3f629e9c0cb722e0a | /home/manikm/.cache/huggingface/hub/models--microsoft--Phi-4-mini-reasoning/snapshots/0e3b1e2d02ee478a3743abe3f629e9c0cb722e0a |
| microsoft/Phi-4-reasoning | 1de18ec97600877ce63dbf60c73b998da99f0195 | /home/manikm/.cache/huggingface/hub/models--microsoft--Phi-4-reasoning/snapshots/1de18ec97600877ce63dbf60c73b998da99f0195 |

## Software package manifest

| package | version |
| --- | --- |
| python | 3.13.11 |
| torch | 2.9.1 |
| transformers | 4.57.6 |
| tokenizers | 0.22.2 |
| datasets | 4.8.4 |
| accelerate | 1.12.0 |
| pandas | 3.0.0 |
| numpy | 2.2.6 |

## Missing/read issues

- core missing/read issues: []
- larger missing/read issues: []