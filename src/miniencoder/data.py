"""Custom dataset, padding, and deterministic stratified splitting."""

from dataclasses import dataclass

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def stratified_split(texts, labels, validation_fraction=0.1, seed=42):
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        list(texts), list(labels), test_size=validation_fraction, random_state=seed, stratify=labels
    )
    return train_texts, val_texts, train_labels, val_labels


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, vocabulary, max_length):
        self.texts, self.labels = list(texts), [int(label) for label in labels]
        if len(self.texts) != len(self.labels):
            raise ValueError("texts and labels must have the same length.")
        if type(max_length) is not int or max_length < 1:
            raise ValueError("max_length must be a positive integer.")
        if any(label not in (0, 1) for label in self.labels):
            raise ValueError("Labels must be 0 or 1.")
        self.tokenizer, self.vocabulary, self.max_length = tokenizer, vocabulary, max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        ids = self.vocabulary.encode(self.tokenizer.tokenize(self.texts[index]))
        ids = [self.vocabulary.cls_id] + ids[: self.max_length - 1]
        mask = [True] * len(ids)
        ids += [self.vocabulary.pad_id] * (self.max_length - len(ids))
        mask += [False] * (self.max_length - len(mask))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.bool),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
            "text": self.texts[index],
        }


def collate_fn(batch):
    return {
        key: torch.stack([item[key] for item in batch])
        if key != "text"
        else [item[key] for item in batch]
        for key in batch[0]
    }


@dataclass
class DatasetStatistics:
    split_size: int
    vocabulary_size: int
    average_length: float
    median_length: float
    truncation_percentage: float
    unknown_percentage: float
