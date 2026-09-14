"""Configuration validation shared by the CLI and model factory."""

import math

import yaml


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        return validate_config(yaml.safe_load(handle))


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping, not empty YAML.")
    allowed = {"seed", "model", "training"}
    if set(config) - allowed:
        raise ValueError(f"Unknown configuration keys: {sorted(set(config) - allowed)}")
    model = config.get("model", {})
    training = config.get("training", {})
    if not isinstance(model, dict) or not isinstance(training, dict):
        raise ValueError("model and training must be mappings.")
    model_keys = {
        "type",
        "embedding_dimension",
        "hidden_dimension",
        "heads",
        "layers",
        "feedforward_dimension",
        "max_length",
        "pooling",
        "positional_encoding",
        "attention_backend",
        "dropout",
    }
    training_keys = {
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "device",
        "gradient_clip_norm",
        "threads",
        "selection_metric",
    }
    for section, keys in ((model, model_keys), (training, training_keys)):
        if set(section) - keys:
            raise ValueError(f"Unknown configuration keys: {sorted(set(section) - keys)}")
    if training.get("selection_metric", "loss") not in {"loss", "accuracy"}:
        raise ValueError("selection_metric must be loss or accuracy.")
    if model.get("type", "transformer") not in {"transformer", "baseline"}:
        raise ValueError("model.type must be transformer or baseline.")
    for key in (
        "embedding_dimension",
        "hidden_dimension",
        "heads",
        "layers",
        "feedforward_dimension",
        "max_length",
    ):
        if key in model and (type(model[key]) is not int or model[key] < 1):
            raise ValueError(f"model.{key} must be a positive integer.")
    for key in ("epochs", "batch_size", "threads"):
        if key in training and (type(training[key]) is not int or training[key] < 1):
            raise ValueError(f"training.{key} must be a positive integer.")
    if type(config.get("seed", 42)) is not int or not 0 <= config.get("seed", 42) < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    dropout = model.get("dropout", 0.1)
    if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1).")
    for key, default in (
        ("learning_rate", 3e-4),
        ("weight_decay", 1e-2),
        ("gradient_clip_norm", 1.0),
    ):
        value = training.get(key, default)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (key != "weight_decay" and value == 0)
        ):
            raise ValueError(f"training.{key} is invalid.")
    if model.get("type", "transformer") == "transformer":
        if model.get("embedding_dimension", 64) % model.get("heads", 4):
            raise ValueError("Embedding dimension must be divisible by heads.")
        if model.get("pooling", "cls") not in {"cls", "mean"}:
            raise ValueError("pooling must be cls or mean.")
        if model.get("attention_backend", "manual") not in {"manual", "sdpa"}:
            raise ValueError("attention_backend must be manual or sdpa.")
        if type(model.get("positional_encoding", True)) is not bool:
            raise ValueError("positional_encoding must be boolean.")
    return config
