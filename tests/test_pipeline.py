import json
import sys

import pytest

from miniencoder.baselines import MeanEmbeddingClassifier
from miniencoder.checkpoint import save_checkpoint
from miniencoder.evaluate import evaluate_checkpoint
from miniencoder.pipeline import read_prepared_data
from miniencoder.predict import main
from miniencoder.tokenizer import RegexTokenizer, build_vocabulary


@pytest.fixture
def artifacts(tmp_path):
    vocabulary = build_vocabulary(["good movie", "bad movie"], RegexTokenizer())
    vocabulary.save(tmp_path / "vocabulary.json")
    (tmp_path / "metadata.json").write_text(json.dumps({"max_length": 8, "lowercase": True}))
    save_checkpoint(tmp_path / "model.pt", MeanEmbeddingClassifier(len(vocabulary.token_to_id)))
    return tmp_path


def test_prediction_without_dataset_splits(artifacts, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["predict", "good movie", "--checkpoint", str(artifacts / "model.pt"),
                                     "--data-dir", str(artifacts)])
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
