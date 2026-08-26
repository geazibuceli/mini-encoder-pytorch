import argparse
import json
from pathlib import Path

from .metrics import classification_metrics, mean_confidence
from .pipeline import load_model, make_loader, read_prepared_data
from .train import run_epoch


def evaluate_checkpoint(checkpoint, data_dir, output, batch_size=32, device="cpu"):
    metadata, vocabulary, splits = read_prepared_data(data_dir)
    model, _ = load_model(checkpoint, vocabulary, device)
    loader = make_loader(splits["test"], vocabulary, metadata, batch_size)
    result = run_epoch(model, loader, None, device)
    report = classification_metrics(result["labels"], result["predictions"])
    report["evaluation_loss"] = result["loss"]
    report["average_confidence"] = mean_confidence(result["logits"].numpy())
    report["class_distribution"] = {str(label): result["labels"].count(label) for label in sorted(set(result["labels"]))}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate a MiniEncoder checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="results/evaluation.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(evaluate_checkpoint(args.checkpoint, args.data_dir, args.output, device=args.device), indent=2))


if __name__ == "__main__": main()
