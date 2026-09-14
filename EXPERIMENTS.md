# Experiment Report

## Objective and Protocol

Improve the sentiment classifier while retaining an inspectable PyTorch encoder implementation. All new models were trained from scratch without pretrained weights or extra labeled data.

- Training: 22,500 reviews; validation: 2,500; test: 25,000.
- The existing split membership and training-only vocabulary were preserved.
- Model changes and seed comparisons used validation results only.
- Initial experiments selected the lowest validation loss. After reviewing their histories, an explicit accuracy-selection option was added and the seed-43 512-token configuration was retrained to optimize the requested metric.
- The final Transformer and fixed baseline were each evaluated once on the test split. There was no tuning after seeing these test results.
- An initial CPU run was interrupted to install CUDA and is excluded from completed-run comparisons.

## Completed Validation Experiments

Values below describe the checkpoint selected in each run, not necessarily the final epoch.

| Configuration | Seed | Tokens | Epochs run | Selected epoch | Selection | Validation accuracy | Validation loss |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `improved_transformer.yaml` | 42 | 128 | 8 | 6 | Loss | 82.16% | 0.4291 |
| `improved_transformer_256.yaml` | 42 | 256 | 10 | 4 | Loss | 86.16% | 0.3606 |
| `improved_transformer_512.yaml` | 42 | 512 | 12 | 6 | Loss | 88.04% | 0.3591 |
| `improved_transformer_512_seed43.yaml` | 43 | 512 | 12 | 4 | Loss | 88.04% | 0.3411 |
| `improved_baseline_512.yaml` | 42 | 512 | 12 | 3 | Loss | 87.80% | 0.3038 |
| `recommended.yaml` | 43 | 512 | 12 | 9 | Accuracy | 89.20% | 0.3763 |

All configurations are in [configs](configs). Complete per-epoch histories and artifact paths are recorded in [results/experiment_summary.json](results/experiment_summary.json). The recommended checkpoint records both its selected epoch's `validation_loss` and the lowest loss observed up to that checkpoint (`best_validation_loss`); they differ when selecting by accuracy.

## Held-Out Test Results

| Model | Accuracy | Precision | Recall | F1 | Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recommended Transformer | 86.684% | 0.9073 | 0.8171 | 0.8599 | 0.4330 |
| Retuned baseline | 86.500% | 0.8673 | 0.8618 | 0.8646 | 0.3200 |

The recommended Transformer correctly classified 21,671 of 25,000 reviews. It improves by 18.724 percentage points over the historical Transformer report (67.96%). The retuned baseline remains competitive and achieves better F1 and loss. The 0.184-point accuracy difference is not evidence of a robust architecture advantage.

The Transformer has lower positive-class recall and high average softmax confidence (94.73%). Confidence is not calibrated. Further work should use a fresh validation protocol for calibration or threshold selection, rather than tuning against this test report.

## Why Longer Inputs Were Tested

Training reviews contain a median of 214 tokens including CLS. The fraction truncated is 87.46% at 128 tokens, 39.57% at 256, and 12.46% at 512. The final encoder also has more parameters than the original; this was an improvement experiment, not an ablation isolating sequence length.

## Environment

- Windows 11, Python 3.14.0.
- NVIDIA GeForce RTX 5080 (16 GB), driver 610.74.
- PyTorch 2.13.0+cu132, CUDA runtime 13.2.
- No mixed precision or pretrained model was used.
- Training seed: 43 for the recommended model; dataset split seed: 42.

Exact installed package versions, split SHA-256 hashes, and the exported model hash are recorded in the experiment summary. GPU execution can produce small numerical differences across runs, even with the same seed; equivalent results across devices or library versions are not guaranteed.

## Reproduction

Follow the [recommended-model commands in the README](README.md#reproduce-the-recommended-model). For the retuned baseline, use `configs/improved_baseline_512.yaml` with the same `data/processed_512` directory. The 256-token configuration requires a prepared copy created with `--max-length 256`; the 128-token configuration uses the original prepared directory.

The compact [inference bundle](results/recommended) is sufficient for prediction and contains no training or test reviews. The large training checkpoints remain local and can be regenerated using the checked-in configurations. The original checkpoints and evaluation reports were preserved.
