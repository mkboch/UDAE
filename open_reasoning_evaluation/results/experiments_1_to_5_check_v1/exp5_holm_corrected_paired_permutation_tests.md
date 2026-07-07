| model_A         | strategy_A   | model_B         | strategy_B   |   weighted_delta_A_minus_B |   raw_two_sided_p |   holm_adjusted_p | holm_significant_at_0.05   |
|:----------------|:-------------|:----------------|:-------------|---------------------------:|------------------:|------------------:|:---------------------------|
| gemma_4_26b_a4b | zero_shot    | gemma_4_e4b     | few_shot_cot |                0.0331933   |        0.0223978  |        0.089591   | False                      |
| gemma_4_26b_a4b | zero_shot    | gemma_4_e4b     | cot          |                0.0352941   |        0.0151985  |        0.0759924  | False                      |
| gemma_4_26b_a4b | zero_shot    | gemma_4_e4b     | zero_shot    |                0.0357143   |        0.0121988  |        0.0731927  | False                      |
| gemma_4_26b_a4b | zero_shot    | gemma_4_26b_a4b | cot          |                0.0378151   |        0.00139986 |        0.00979902 | True                       |
| gemma_4_e4b     | few_shot_cot | gemma_4_e4b     | cot          |                0.00210084  |        0.877712   |        1          | False                      |
| gemma_4_e4b     | few_shot_cot | gemma_4_e4b     | zero_shot    |                0.00252101  |        0.872613   |        1          | False                      |
| gemma_4_e4b     | cot          | gemma_4_e4b     | zero_shot    |                0.000420168 |        0.967303   |        1          | False                      |