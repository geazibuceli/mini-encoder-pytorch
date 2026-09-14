import json
import sys

import pytest

from miniencoder.baselines import MeanEmbeddingClassifier
from miniencoder.checkpoint import save_checkpoint
from miniencoder.evaluate import evaluate_checkpoint
from miniencoder.pipeline import load_model, read_prepared_data
from miniencoder.predict import main
from miniencoder.tokenizer import RegexTokenizer, Vocabulary, build_vocabulary


@pytest.fixture
def artifacts(tmp_path):
    vocabulary = build_vocabulary(["good movie", "bad movie"], RegexTokenizer())
    vocabulary.save(tmp_path / "vocabulary.json")
    (tmp_path / "metadata.json").write_text(json.dumps({"max_length": 8, "lowercase": True}))
    save_checkpoint(tmp_path / "model.pt", MeanEmbeddingClassifier(len(vocabulary.token_to_id)))
    return tmp_path


def test_prediction_without_dataset_splits(artifacts, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict",
            "good movie",
            "--checkpoint",
            str(artifacts / "model.pt"),
            "--data-dir",
            str(artifacts),
        ],
    )
    main()
    assert "Prediction:" in capsys.readouterr().out


def test_evaluation_with_only_test_split(artifacts):
    records = [{"text": "good movie", "label": 1}, {"text": "bad movie", "label": 0}]
    (artifacts / "test.jsonl").write_text("\n".join(json.dumps(row) for row in records))
    report = evaluate_checkpoint(artifacts / "model.pt", artifacts, artifacts / "report.json")
    assert report["num_examples"] == 2
    assert json.loads((artifacts / "report.json").read_text()) == report


def test_default_loads_all_splits(artifacts):
    for split in ("train", "validation", "test"):
        (artifacts / f"{split}.jsonl").write_text('{"text": "good movie", "label": 1}\n')
    _, _, splits = read_prepared_data(artifacts)
    assert set(splits) == {"train", "validation", "test"}
    assert all(len(rows) == 1 for rows in splits.values())


def test_unknown_split_rejected(artifacts):
    with pytest.raises(ValueError, match="Unknown dataset split"):
        read_prepared_data(artifacts, split_names=("missing",))


def test_checkpoint_rejects_mismatched_preprocessing(artifacts):
    vocabulary = Vocabulary.load(artifacts / "vocabulary.json")
    path = artifacts / "verified.pt"
    save_checkpoint(
        path,
        MeanEmbeddingClassifier(len(vocabulary.token_to_id)),
        vocabulary_metadata={
            "vocabulary_sha256": vocabulary.fingerprint(),
            "lowercase": True,
            "max_length": 8,
        },
    )
    swapped = dict(vocabulary.token_to_id)
    swapped["good"], swapped["bad"] = swapped["bad"], swapped["good"]
    with pytest.raises(ValueError, match="Vocabulary"):
        load_model(path, Vocabulary(swapped))
    with pytest.raises(ValueError, match="lowercase"):
        load_model(path, vocabulary, metadata={"max_length": 8, "lowercase": False})
    with pytest.raises(ValueError, match="max_length"):
        load_model(path, vocabulary, metadata={"max_length": 16, "lowercase": True})
    load_model(path, vocabulary, metadata={"max_length": 8, "lowercase": True})


def test_export_bundle_without_dataset(artifacts):
    import torch

    from miniencoder.export import export_checkpoint

    output = artifacts / "bundle"
    checkpoint = export_checkpoint(artifacts / "model.pt", artifacts, output)
    metadata, vocabulary, splits = read_prepared_data(output, split_names=())
    original, _ = load_model(artifacts / "model.pt", vocabulary)
    restored, payload = load_model(checkpoint, vocabulary, metadata=metadata)
    ids = torch.tensor([[2, 3, 4]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    torch.testing.assert_close(original(ids, mask), restored(ids, mask))
    assert not splits
    assert payload["optimizer_state"] is None
    assert payload["vocabulary_metadata"]["vocabulary_sha256"] == vocabulary.fingerprint()
