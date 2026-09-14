# MiniEncoder PyTorch

An educational sentiment classifier for IMDB, featuring a Transformer and a mean-embedding baseline.

## Installation

Run the following commands from this directory using Python 3.10 or later:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev,data,visualization]"
```

## Usage

```powershell
.venv/Scripts/python.exe -m miniencoder.doctor
.venv/Scripts/python.exe -m miniencoder.prepare_data
.venv/Scripts/python.exe -m miniencoder.train --config configs/quick_test.yaml --checkpoint results/model.pt
.venv/Scripts/python.exe -m miniencoder.evaluate --checkpoint results/model.pt
.venv/Scripts/python.exe -m miniencoder.predict "A wonderful movie" --checkpoint results/model.pt
.venv/Scripts/python.exe -m pytest -q
```

Existing prepared data can be used without running `prepare_data`, which downloads IMDB.
The model's sequence capacity is inferred from the data metadata when `model.max_length` is omitted.
An explicitly configured capacity smaller than the prepared sequence length is rejected before training.
The checkpoint stores the epoch with the lowest validation loss, including the model configuration and optimizer and scheduler states.
Its training history ends at the selected epoch; the CLI prints the results of the final training epoch.

When training through the API, call `set_seed(seed)` before constructing the model and pass the same seed to `train_model`.
`train_model` does not reinitialize existing model weights. The model in memory retains the final epoch's weights;
load the saved checkpoint to use the best epoch. Reproducibility across different devices and versions is not guaranteed.

## Recovered Project Structure

The maintained source code lives in `src/miniencoder`. It was recovered from the existing copy in `build/lib`.
The tests in `tests` are new regression tests; the original test source files were missing.
`build/lib` and the old caches were preserved and should not be used as development sources.
Existing checkpoints and reports were preserved; these fixes do not automatically retrain those models.
