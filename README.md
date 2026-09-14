# MiniEncoder PyTorch

A compact Transformer encoder for sentiment classification on IMDB, built to make attention, training, and evaluation easy to inspect.

Compare manual multi-head attention with PyTorch scaled dot-product attention (SDPA), visualize attention weights, and measure the encoder against a mean-embedding baseline.

## Features

- Regex tokenization and a deterministic vocabulary built from training data.
- Transformer blocks with pre-layer normalization, residual connections, and optional sinusoidal positions.
- CLS or masked mean pooling for binary sentiment classification.
- AdamW training with gradient clipping and cosine learning-rate scheduling.
- Best-validation checkpoints containing model configuration and optimizer state.
- Evaluation reports, CPU benchmarks, and attention heatmaps.

## Installation

Use Python 3.10 or later. Run these commands from the directory containing `pyproject.toml`:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev,data,visualization]"
```

On Linux or macOS, replace `.venv/Scripts/python.exe` with `.venv/bin/python` throughout these instructions.

For inference and training with prepared data, use `pip install -e .` inside the virtual environment. The `data`, `visualization`, and `dev` extras add dataset downloading, plotting, and testing dependencies respectively.

## Quick Start

When the included checkpoint and prepared artifacts are available, no download or training is needed:

```powershell
.venv/Scripts/python.exe -m miniencoder.doctor
.venv/Scripts/python.exe -m miniencoder.predict "A wonderful movie" --checkpoint results/real_data_model.pt
```

Prediction prints the label, softmax confidence, model type, and attention backend. Inference needs the checkpoint and its matching `vocabulary.json` and `metadata.json` in `data/processed` (or a directory passed with `--data-dir`). It does not load any dataset splits.

## Train and Evaluate

### Prepare IMDB

Skip this step when using the existing prepared dataset. Running it downloads IMDB and writes prepared artifacts to the output directory.

```powershell
.venv/Scripts/python.exe -m miniencoder.prepare_data --output data/processed --max-length 128
```

The default split uses 22,500 training reviews, 2,500 validation reviews, and the official 25,000-review test set. The vocabulary uses only training reviews. Labels are `0` (negative) and `1` (positive).

### Train

```powershell
.venv/Scripts/python.exe -m miniencoder.train --config configs/quick_test.yaml --checkpoint results/model.pt
```

The supplied configuration runs one epoch over the complete prepared training split, not a tiny smoke test. Training time depends on the device and dataset size. Automatic device selection prefers CUDA, then MPS, then CPU.

The checkpoint stores the epoch with the lowest validation loss. Its history ends at that epoch; the CLI prints the final epoch's results. An existing checkpoint at the selected path is replaced when a new best epoch is saved.

### Evaluate

```powershell
.venv/Scripts/python.exe -m miniencoder.evaluate --checkpoint results/model.pt --output results/evaluation.json
```

Evaluation loads only the test split and writes accuracy, precision, recall, F1, a confusion matrix, cross-entropy loss, and average softmax confidence.

## Configuration

Edit [configs/quick_test.yaml](configs/quick_test.yaml) to run an experiment.

| Setting | Supplied value | Purpose |
| --- | --- | --- |
| `seed` | `42` | Applied before model construction in the CLI |
| `model.type` | `transformer` | Use `baseline` for the mean-embedding classifier |
| `model.embedding_dimension` | `64` | Embedding width; must be divisible by the Transformer head count |
| `model.heads` / `model.layers` | `4` / `2` | Attention heads and encoder blocks |
| `model.feedforward_dimension` | `128` | Feed-forward network width |
| `model.pooling` | `cls` | `cls` or masked `mean` pooling |
| `model.attention_backend` | `manual` | `manual` or `sdpa` |
| `model.max_length` | Inferred from data | Positional capacity, including CLS |
| `training.epochs` / `training.batch_size` | `1` / `32` | Training duration and batch size |
| `training.learning_rate` | `0.0003` | AdamW learning rate |
| `training.device` | `auto` | Automatic selection or an explicit device such as `cpu` |

Transformer-specific settings do not apply to the baseline. An explicitly configured positional capacity smaller than the prepared sequence length is rejected before training.

For the Python API, call `set_seed(seed)` before building a model and pass the same seed to `train_model`. That function preserves existing weights and leaves the model in memory at its final epoch; load the checkpoint to use the best epoch. Reproducibility across devices and library versions is not guaranteed.

## Recorded Results

These existing reports cover 25,000 test reviews. They are not a new evaluation of the latest code.

| Model | Accuracy | Precision | Recall | F1 | Evaluation loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Transformer](results/real_data_evaluation.json) | 67.96% | 0.6883 | 0.6566 | 0.6720 | 0.6053 |
| [Mean-embedding baseline](results/baseline_real_data_evaluation.json) | 75.99% | 0.7607 | 0.7584 | 0.7596 | 0.4964 |

The baseline performed better in these runs. This does not establish a general advantage for either architecture. Existing checkpoints were preserved and have not been retrained after the training fixes.

```powershell
.venv/Scripts/python.exe -m miniencoder.compare results/real_data_evaluation.json results/baseline_real_data_evaluation.json
```

## Benchmark and Visualize

Measure CPU forward latency and throughput, with JSON and Markdown output:

```powershell
.venv/Scripts/python.exe -m miniencoder.benchmark --checkpoint results/real_data_model.pt --iterations 10 --output results/benchmark.json
```

Generate a heatmap for a manual-attention Transformer checkpoint:

```powershell
.venv/Scripts/python.exe -m miniencoder.visualize_attention "A wonderful movie" --checkpoint results/real_data_model.pt --layer 0 --head 0 --output results/attention.png
```

SDPA does not expose attention weights through this implementation. The baseline has no attention layers.

## Development

```powershell
.venv/Scripts/python.exe -m pytest -q
```

Regression tests cover checkpoint loading, best-epoch selection, reproducible CPU training, sequence capacity, and selective dataset loading.

```text
configs/            Experiment configuration
src/miniencoder/    Maintained Python package
tests/             Regression tests
data/processed/     Vocabulary, metadata, and dataset splits
results/            Checkpoints and experiment reports
```

The source was recovered from an existing `build/lib` copy; the original test sources were missing. Old build files and caches are preserved artifacts. Make development changes in `src/miniencoder`.

## Scope and Limitations

This is an educational English movie-review classifier. It has not been validated for other languages or domains, and softmax confidence is not a calibrated probability of correctness. Use matching preprocessing artifacts with each checkpoint: a different vocabulary can change token meanings even when its size matches.
