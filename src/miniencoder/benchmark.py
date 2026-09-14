import argparse
import json
import time
from pathlib import Path

import torch

from .pipeline import load_model, make_loader, read_prepared_data


def benchmark_model(model, batch, iterations=10, warmup=3, device="cpu"):
    if type(iterations) is not int or iterations < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError("iterations must be positive and warmup must be non-negative integers.")
    device = torch.device(device)
    model.to(device).eval()
    batch = {
        key: value.to(device)
        for key, value in batch.items()
        if key in ("input_ids", "attention_mask")
    }
    with torch.inference_mode():
        for _ in range(warmup):
            model(**batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(iterations):
            model(**batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - start) / iterations
    examples = batch["input_ids"].size(0)
    tokens = batch["input_ids"].numel()
    return {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "forward_latency_seconds": elapsed,
        "examples_per_second": examples / elapsed,
        "tokens_per_second": tokens / elapsed,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark MiniEncoder models.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="results/benchmark.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    metadata, vocabulary, splits = read_prepared_data(args.data_dir, split_names=("test",))
    model, _ = load_model(args.checkpoint, vocabulary, args.device, metadata=metadata)
    batch = next(iter(make_loader(splits["test"], vocabulary, metadata, args.batch_size)))
    result = benchmark_model(model, batch, args.iterations, device=args.device)
    result["checkpoint_size_bytes"] = Path(args.checkpoint).stat().st_size
    result["device"] = args.device
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(
        "# Benchmark\n\n| Metric | Value |\n|---|---:|\n"
        + "\n".join(f"| {key} | {value} |" for key, value in result.items()),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
