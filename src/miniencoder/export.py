"""Export a compact inference bundle with matching preprocessing artifacts."""

import argparse
import json
from pathlib import Path

from .checkpoint import save_checkpoint
from .pipeline import load_model, read_prepared_data


def export_checkpoint(checkpoint, data_dir, output):
    metadata, vocabulary, _ = read_prepared_data(data_dir, split_names=())
    model, payload = load_model(checkpoint, vocabulary, metadata=metadata)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata, vocabulary_sha256=vocabulary.fingerprint(), tokenizer="regex-v1")
    save_checkpoint(
        output / "model.pt",
        model,
        vocabulary_metadata=metadata,
        epoch=payload.get("epoch"),
        random_seed=payload.get("random_seed"),
        best_validation_loss=payload.get("best_validation_loss"),
        validation_loss=payload.get("validation_loss"),
        validation_accuracy=payload.get("validation_accuracy"),
        selection_metric=payload.get("selection_metric", "loss"),
        selection_score=payload.get("selection_score", payload.get("best_validation_loss")),
        training_config=payload.get("training_config"),
        runtime=payload.get("runtime"),
    )
    vocabulary.save(output / "vocabulary.json")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output / "model.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(export_checkpoint(args.checkpoint, args.data_dir, args.output))


if __name__ == "__main__":
    main()
