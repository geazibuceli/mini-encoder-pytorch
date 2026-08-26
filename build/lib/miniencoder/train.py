"""Manual training loop and training CLI."""

import argparse
import json
import math
from pathlib import Path

import torch

from .baselines import MeanEmbeddingClassifier
from .checkpoint import save_checkpoint
from .config import load_config
from .model import TransformerClassifier
from .utils import select_device, set_seed


def build_model(config, vocabulary_size, padding_id=0):
    model_config = config.get("model", config)
    if model_config.get("type", "transformer") == "baseline":
        return MeanEmbeddingClassifier(vocabulary_size, model_config.get("embedding_dimension", 64),
                                       model_config.get("hidden_dimension", 64), padding_id,
                                       model_config.get("dropout", 0.1))
    return TransformerClassifier(vocabulary_size, padding_id=padding_id, **{
        key: model_config[key] for key in ("embedding_dimension", "heads", "layers", "feedforward_dimension",
                                           "max_length", "pooling", "positional_encoding", "attention_backend", "dropout")
        if key in model_config})


def run_epoch(model, loader, optimizer=None, device="cpu", clip_norm=1.0):
    training = optimizer is not None
    model.train(training)
    total_loss, predictions, labels, logits_history = 0.0, [], [], []
    criterion = torch.nn.CrossEntropyLoss()
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            inputs = {key: batch[key].to(device) for key in ("input_ids", "attention_mask", "label")}
            logits = model(inputs["input_ids"], inputs["attention_mask"])
            loss = criterion(logits, inputs["label"])
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            total_loss += loss.item() * len(inputs["label"])
            logits_history.append(logits.detach().cpu())
            predictions.extend(logits.argmax(1).cpu().tolist())
            labels.extend(inputs["label"].cpu().tolist())
        return {"loss": total_loss / max(1, len(labels)), "predictions": predictions, "labels": labels,
            "logits": torch.cat(logits_history) if logits_history else torch.empty((0, 2))}


def train_model(model, train_loader, validation_loader, epochs=1, learning_rate=3e-4, weight_decay=1e-2,
                device="cpu", clip_norm=1.0, checkpoint_path=None, seed=42):
    set_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, epochs))
    history, best_loss = [], math.inf
    for epoch in range(1, epochs + 1):
        train_result = run_epoch(model, train_loader, optimizer, device, clip_norm)
        validation_result = run_epoch(model, validation_loader, None, device, clip_norm)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": train_result["loss"], "validation_loss": validation_result["loss"]}
        history.append(record)
        if validation_result["loss"] < best_loss:
            best_loss = validation_result["loss"]
            if checkpoint_path:
                save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch=epoch, global_step=epoch,
                                best_validation_loss=best_loss, training_history=history, random_seed=seed)
    return history


def main():
    parser = argparse.ArgumentParser(description="Train a MiniEncoder model.")
    parser.add_argument("--config", default="configs/quick_test.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint", default="results/model.pt")
    args = parser.parse_args()
    config = load_config(args.config)
    from .pipeline import make_loader, read_prepared_data
    metadata, vocabulary, splits = read_prepared_data(args.data_dir)
    model = build_model(config, len(vocabulary.token_to_id), vocabulary.pad_id)
    train_config = config.get("training", {})
    train_loader = make_loader(splits["train"], vocabulary, metadata, train_config.get("batch_size", 32), True)
    validation_loader = make_loader(splits["validation"], vocabulary, metadata, train_config.get("batch_size", 32))
    history = train_model(model, train_loader, validation_loader, train_config.get("epochs", 1),
                          train_config.get("learning_rate", 3e-4), train_config.get("weight_decay", 1e-2),
                          str(select_device(train_config.get("device", "auto"))),
                          train_config.get("gradient_clip_norm", 1.0), None, config.get("seed", 42))
    from .checkpoint import save_checkpoint
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.checkpoint, model, epoch=len(history), training_history=history,
                    best_validation_loss=history[-1]["validation_loss"], random_seed=config.get("seed", 42),
                    model_config=config["model"], vocabulary_metadata=metadata)
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__": main()
