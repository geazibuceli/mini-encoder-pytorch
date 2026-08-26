import argparse
import json
from pathlib import Path

from .data import stratified_split
from .tokenizer import RegexTokenizer, build_vocabulary


def main():
    parser = argparse.ArgumentParser(description="Download and prepare IMDB data.")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("Install the datasets package before preparing IMDB.") from error
    dataset = load_dataset("stanfordnlp/imdb")
    texts, labels = dataset["train"]["text"], dataset["train"]["label"]
    train_texts, validation_texts, train_labels, validation_labels = stratified_split(
        texts, labels, args.validation_fraction, args.seed)
    tokenizer = RegexTokenizer()
    vocabulary = build_vocabulary(train_texts, tokenizer, args.min_frequency)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    vocabulary.save(output / "vocabulary.json")
    splits = {"train": zip(train_texts, train_labels), "validation": zip(validation_texts, validation_labels),
              "test": zip(dataset["test"]["text"], dataset["test"]["label"])}
    for name, records in splits.items():
        (output / f"{name}.jsonl").write_text("".join(json.dumps({"text": text, "label": int(label)}) + "\n"
                                    for text, label in records), encoding="utf-8")
    metadata = {"seed": args.seed, "max_length": args.max_length, "train_size": len(train_texts),
                "validation_size": len(validation_texts), "test_size": len(dataset["test"]),
                "vocabulary_size": len(vocabulary.token_to_id), "lowercase": True}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__": main()
