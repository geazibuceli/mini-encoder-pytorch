import argparse

import torch

from .data import SentimentDataset
from .pipeline import load_model, read_prepared_data
from .tokenizer import RegexTokenizer


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment for a review.")
    parser.add_argument("review")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    metadata, vocabulary, _ = read_prepared_data(args.data_dir)
    model, payload = load_model(args.checkpoint, vocabulary, args.device)
    item = SentimentDataset([args.review], [0], RegexTokenizer(metadata.get("lowercase", True)), vocabulary,
                            metadata["max_length"])[0]
    with torch.inference_mode():
        logits = model(item["input_ids"].unsqueeze(0).to(args.device), item["attention_mask"].unsqueeze(0).to(args.device))
        probabilities = torch.softmax(logits, dim=-1)[0]
    label = int(probabilities.argmax())
    print(f"Prediction: {'Positive' if label else 'Negative'}")
    print(f"Confidence: {probabilities[label].item() * 100:.2f}%")
    print(f"Model: {payload['model_config'].get('type', 'TransformerClassifier')}")
    print(f"Attention backend: {payload['model_config'].get('attention_backend', 'none').upper()}")


if __name__ == "__main__": main()
