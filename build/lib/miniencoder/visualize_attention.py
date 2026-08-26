import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .data import SentimentDataset
from .pipeline import load_model, read_prepared_data
from .tokenizer import RegexTokenizer


def main():
    parser = argparse.ArgumentParser(description="Visualize manual-attention weights.")
    parser.add_argument("review")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="results/attention.png")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=0)
    args = parser.parse_args()
    metadata, vocabulary, _ = read_prepared_data(args.data_dir)
    model, payload = load_model(args.checkpoint, vocabulary, "cpu")
    if payload["model_config"].get("attention_backend") != "manual":
        raise SystemExit("Attention visualization requires a manual-attention checkpoint.")
    item = SentimentDataset([args.review], [0], RegexTokenizer(), vocabulary, metadata["max_length"])[0]
    with torch.inference_mode():
        _, attention = model(item["input_ids"].unsqueeze(0), item["attention_mask"].unsqueeze(0), True)
    length = int(item["attention_mask"].sum())
    tokens = vocabulary.decode(item["input_ids"][:length])
    matrix = attention[args.layer][0, args.head, :length, :length].numpy()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(5, length / 2), max(4, length / 2)))
    plt.imshow(matrix, cmap="viridis")
    plt.xticks(range(length), tokens, rotation=90)
    plt.yticks(range(length), tokens)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved attention heatmap to {args.output}")


if __name__ == "__main__": main()
