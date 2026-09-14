import json
import sys

import pytest
import torch

from miniencoder import train
from miniencoder.baselines import MeanEmbeddingClassifier
from miniencoder.checkpoint import save_checkpoint
from miniencoder.model import TransformerClassifier
from miniencoder.pipeline import load_model
from miniencoder.tokenizer import RegexTokenizer, build_vocabulary


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("model_class", [TransformerClassifier, MeanEmbeddingClassifier])
def test_checkpoint_roundtrip(tmp_path, model_class):
    vocabulary = build_vocabulary(["good movie"], RegexTokenizer())
    model = model_class(len(vocabulary.token_to_id)).eval()
    ids = torch.tensor([[2, 3, 4]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    path = tmp_path / "nested" / "model.pt"
    save_checkpoint(path, model)
    restored, _ = load_model(path, vocabulary)
    torch.testing.assert_close(model(ids, mask), restored(ids, mask))


def test_best_epoch_is_kept(tmp_path, monkeypatch):
    model = MeanEmbeddingClassifier(5)
    losses = iter([0.3, 0.8])

    def epoch(model, loader, optimizer=None, *args):
        if optimizer is not None:
            optimizer.zero_grad()
            model.embedding.weight.sum().backward()
            optimizer.step()
            return {"loss": 1.0}
        return {"loss": next(losses)}

    monkeypatch.setattr(train, "run_epoch", epoch)
    path = tmp_path / "best.pt"
    history = train.train_model(model, [0, 1], [0], epochs=2, checkpoint_path=path)
    payload = torch.load(path, weights_only=True)
    assert len(history) == 2
    assert payload["epoch"] == 1
    assert payload["global_step"] == 2
    assert payload["best_validation_loss"] == 0.3
    assert payload["model_config"]["type"] == "baseline"
    assert not torch.equal(payload["model_state"]["embedding.weight"], model.embedding.weight)


def test_cli_reproducible_and_infers_sequence_length(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    vocabulary = build_vocabulary(["good movie", "bad movie"], RegexTokenizer())
    vocabulary.save(data / "vocabulary.json")
    (data / "metadata.json").write_text(json.dumps({"max_length": 256, "lowercase": True}))
    records = [{"text": "good movie", "label": 1}, {"text": "bad movie", "label": 0}]
    for split in ("train", "validation", "test"):
        (data / f"{split}.jsonl").write_text("\n".join(json.dumps(row) for row in records))
    config = tmp_path / "config.yaml"
    config.write_text("seed: 42\nmodel:\n  embedding_dimension: 8\n  heads: 2\n  layers: 1\n  feedforward_dimension: 16\ntraining:\n  epochs: 2\n  batch_size: 2\n  device: cpu\n")
    payloads = []
    for run in range(2):
        path = tmp_path / f"model{run}.pt"
        monkeypatch.setattr(sys, "argv", ["train", "--config", str(config), "--data-dir", str(data),
                                         "--checkpoint", str(path)])
        train.main()
        restored, payload = load_model(path, vocabulary)
        assert restored.position.encoding.shape[1] == 256
        assert payload["vocabulary_metadata"]["max_length"] == 256
        payloads.append(payload)
    assert payloads[0]["training_history"] == payloads[1]["training_history"]
    for name, tensor in payloads[0]["model_state"].items():
        torch.testing.assert_close(tensor, payloads[1]["model_state"][name], rtol=0, atol=0)
    config.write_text("model:\n  max_length: 128\n")
    with pytest.raises(ValueError, match="prepared data max_length"):
        train.main()


def test_invalid_training_inputs():
    model = MeanEmbeddingClassifier(5)
    with pytest.raises(ValueError, match="epochs"):
        train.train_model(model, [0], [0], epochs=0)
    with pytest.raises(ValueError, match="empty"):
        train.train_model(model, [], [0])
