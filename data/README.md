# Dataset Reconstruction Files

This directory contains the selected identifiers and prepared records associated with the evaluation matrices reported in the paper.

## Matched core

`prepared/` and `indices/` correspond to the 238-example matched protocol for all four benchmarks.

- ARC-Challenge: 238 examples
- GSM8K: 238 examples
- MATH L1--L3: all 238 Level 1--3 test examples
- TruthfulQA MC1: 238 corrected counterbalanced examples

The TruthfulQA files are the corrected choice-balanced records retained from the final follow-up audit. They replace the earlier original-order representation.

## Larger-sample robustness matrix

`larger_sample/prepared/` and `larger_sample/indices/` correspond to:

- ARC-Challenge: 500 examples
- GSM8K: 500 examples
- MATH L1--L3: 238 examples
- TruthfulQA MC1: 500 corrected counterbalanced examples

The ARC-Challenge and GSM8K larger-sample records were reconstructed from the retained final `journal_full500_v1` raw runs and preserve the exact sample identifiers, questions, and gold answers used in those runs. The MATH component is unchanged because all available Level 1--3 test examples total 238. The TruthfulQA larger-sample files are the retained corrected counterbalanced records.

## Preparation script

`prepare_datasets.py` is the retained general dataset-preparation script.

Important: the final reported TruthfulQA results use the corrected counterbalanced TruthfulQA records distributed in this directory. Re-running only the generic preparation script does not substitute for those retained corrected records.
