"""Shared runtime helpers for the command-line workflows."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import SentimentDataset, collate_fn
from .tokenizer import RegexTokenizer, Vocabulary
from .train import build_model


def read_prepared_data(data_dir):
    root = Path(data_dir)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    vocabulary = Vocabulary.load(root / "vocabulary.json")
    splits = {}
    for name in ("train", "validation", "test"):
        path = root / f"{name}.jsonl"
        splits[name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return metadata, vocabulary, splits


def make_loader(records, vocabulary, metadata, batch_size=32, shuffle=False):
    dataset = SentimentDataset([item["text"] for item in records], [item["label"] for item in records],
                               RegexTokenizer(metadata.get("lowercase", True)), vocabulary,
                               metadata["max_length"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def load_model(checkpoint_path, vocabulary, device="cpu"):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = {"model": payload["model_config"]}
    model = build_model(config, len(vocabulary.token_to_id), vocabulary.pad_id).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload
