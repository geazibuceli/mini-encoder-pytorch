import pytest
import torch

from miniencoder.baselines import MeanEmbeddingClassifier
from miniencoder.benchmark import benchmark_model


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_cpu_benchmark(device):
    model = MeanEmbeddingClassifier(5, embedding_dimension=8, hidden_dimension=8)
    batch = {"input_ids": torch.tensor([[2, 3]]), "attention_mask": torch.tensor([[True, True]])}
    report = benchmark_model(model, batch, iterations=2, warmup=1, device=device)
    assert report["forward_latency_seconds"] > 0
    assert report["examples_per_second"] > 0
    assert report["peak_cuda_memory_bytes"] is None
    with pytest.raises(ValueError, match="iterations"):
        benchmark_model(model, batch, iterations=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_indexed_cuda_device_benchmark():
    model = MeanEmbeddingClassifier(5, embedding_dimension=8, hidden_dimension=8)
    batch = {"input_ids": torch.tensor([[2, 3]]), "attention_mask": torch.tensor([[True, True]])}
    report = benchmark_model(model, batch, iterations=2, warmup=1, device="cuda:0")
    assert report["peak_cuda_memory_bytes"] > 0
    assert report["forward_latency_seconds"] > 0
