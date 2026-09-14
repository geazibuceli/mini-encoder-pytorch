# MiniEncoder PyTorch

A compact Transformer encoder for sentiment classification on IMDB, built to make attention, training, and evaluation easy to inspect.

Licensed under the [MIT License](LICENSE).

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

For the RTX 5080 setup verified in this project (Windows, Python 3.14, NVIDIA driver 610.74), install the CUDA build after the base installation:

```powershell
.venv/Scripts/python.exe -m pip install --upgrade torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132
.venv/Scripts/python.exe -m miniencoder.doctor
```

The diagnostic reports the GPU name and performs a CUDA forward/backward check. For other systems, use the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) to select a compatible build.

On Linux or macOS, replace `.venv/Scripts/python.exe` with `.venv/bin/python` throughout these instructions.

For inference and training with prepared data, use `pip install -e .` inside the virtual environment. The `data`, `visualization`, and `dev` extras add dataset downloading, plotting, and testing dependencies respectively.

## Quick Start

The exported Transformer bundle is ready for inference, without downloading IMDB or loading dataset splits:

```powershell
.venv/Scripts/python.exe -m miniencoder.doctor
.venv/Scripts/python.exe -m miniencoder.predict "A wonderful movie" --checkpoint results/recommended/model.pt --data-dir results/recommended
```

Prediction prints the label, softmax confidence, model type, and attention backend. The approximately 25.7 MB bundle contains the model and its matching vocabulary and metadata. Add `--device cuda` to use an NVIDIA GPU; CPU inference also works.

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

By default, the checkpoint stores the epoch with the lowest validation loss. Set `training.selection_metric: accuracy` to select the highest validation accuracy instead (ties keep the earlier epoch). Its embedded history ends at the selected epoch; the CLI writes the complete history to `<checkpoint>.history.json` and prints progress after every epoch. An existing checkpoint at the selected path is replaced when a new best epoch is saved.

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
| `training.selection_metric` | `loss` (implicit) | Select by lowest loss or highest `accuracy` on validation |
| `training.threads` | Up to `4` (implicit) | CPU threads used by PyTorch |

Transformer-specific settings do not apply to the baseline. An explicitly configured positional capacity smaller than the prepared sequence length is rejected before training.

For the Python API, call `set_seed(seed)` before building a model and pass the same seed to `train_model`. That function preserves existing weights and leaves the model in memory at its final epoch; load the checkpoint to use the best epoch. Reproducibility across devices and library versions is not guaranteed.

## Results

The new models were trained from scratch on 22,500 reviews, using 2,500 validation reviews for experiment and checkpoint selection. The final Transformer and the retuned baseline were each evaluated once on the 25,000-review test set after selection. No parameters were changed after test evaluation.

| Model | Test accuracy | Test F1 | Test loss | Status |
| --- | ---: | ---: | ---: | --- |
| [Original Transformer](results/real_data_evaluation.json) | 67.96% | 0.6720 | 0.6053 | Historical report |
| [Original baseline](results/baseline_real_data_evaluation.json) | 75.99% | 0.7596 | 0.4964 | Historical report |
| [Retuned baseline, 512 tokens](results/improved_baseline_evaluation.json) | 86.50% | 0.8646 | 0.3200 | New evaluation |
| [Recommended Transformer, 512 tokens](results/recommended_evaluation.json) | **86.68%** | 0.8599 | 0.4330 | New evaluation |

The Transformer improves on its historical report by **18.72 percentage points**. Its 0.18-point accuracy advantage over the retuned baseline is small; the baseline has better F1 and loss, so these runs do not demonstrate an overall advantage for the Transformer.

The recommended Transformer uses 128-dimensional embeddings, two encoder layers, masked mean pooling, SDPA, and dropout 0.3. Its selected checkpoint is epoch 9 of a 12-epoch run with seed 43, selected by validation accuracy (**89.20%**). Validation accuracy is not the test accuracy.

The original 128-token limit truncated 87.46% of training reviews, compared with 12.46% at 512 tokens. Longer inputs, a larger model, and more training changed together, so the gain cannot be attributed to a single change.

See [EXPERIMENTS.md](EXPERIMENTS.md) for the complete validation comparison, reproduction commands, environment, and limitations. Machine-readable results and dataset checksums are in [experiment_summary.json](results/experiment_summary.json).

## Reproduce the Recommended Model

Reuse the existing raw review text and vocabulary with a longer sequence limit. This preserves split membership and leaves `data/processed` unchanged:

```powershell
.venv/Scripts/python.exe -m miniencoder.prepare_data --source-dir data/processed --output data/processed_512 --max-length 512
.venv/Scripts/python.exe -m miniencoder.train --config configs/recommended.yaml --data-dir data/processed_512 --checkpoint results/improved_recommended.pt
.venv/Scripts/python.exe -m miniencoder.evaluate --checkpoint results/improved_recommended.pt --data-dir data/processed_512 --device cuda --output results/recommended_evaluation.json
.venv/Scripts/python.exe -m miniencoder.export --checkpoint results/improved_recommended.pt --data-dir data/processed_512 --output results/recommended
```

The export strips optimizer and scheduler states and includes the matching preprocessing files. Generated dataset copies and full experiment checkpoints are ignored by Git; the compact bundle, configurations, reports, and training histories are versionable. These commands overwrite the selected output files.

## Benchmark and Visualize

Measure forward latency and throughput, with JSON and Markdown output. CPU is the default; pass `--device cuda:0` to benchmark the GPU:

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
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/python.exe -m ruff format --check src tests
```

Regression tests cover attention outputs and gradients, padding invariance, tokenizer and position behavior, configuration validation, checkpoint compatibility, inference exports, best-epoch selection, reproducible CPU training, selective dataset loading, and GPU benchmarking when CUDA is available. GitHub Actions runs CPU tests and Ruff checks on Python 3.10 and 3.12.

```text
configs/            Experiment configuration
src/miniencoder/    Maintained Python package
tests/              Regression tests
data/processed/     Vocabulary, metadata, and dataset splits
results/            Checkpoints and experiment reports
```

The source was recovered from an existing `build/lib` copy; the original test sources were missing. Old build files and caches have been removed from version control and are ignored locally. Make development changes in `src/miniencoder`. Full experiment checkpoints are local artifacts; a compact inference bundle is exported separately.

## Scope and Limitations

This is an educational English movie-review classifier. It has not been validated for other languages or domains, and softmax confidence is not a calibrated probability of correctness. New CLI checkpoints validate a SHA-256 fingerprint of the vocabulary and the preprocessing settings. Legacy checkpoints remain loadable, but their original token mapping cannot be verified because they lack a fingerprint.
