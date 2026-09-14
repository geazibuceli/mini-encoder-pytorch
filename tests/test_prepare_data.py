import json
import sys

import pytest

from miniencoder.prepare_data import main


def test_reuse_data_preserves_splits(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    (source / "metadata.json").write_text(json.dumps({"max_length": 128, "seed": 42}))
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "vocabulary.json"):
        (source / name).write_text('{"preserved": true}')
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare", "--source-dir", str(source), "--output", str(output), "--max-length", "256"],
    )
    main()
    assert json.loads((output / "metadata.json").read_text())["max_length"] == 256
    assert json.loads((source / "metadata.json").read_text())["max_length"] == 128
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "vocabulary.json"):
        assert (source / name).read_bytes() == (output / name).read_bytes()
    monkeypatch.setattr(
        sys, "argv", ["prepare", "--source-dir", str(source), "--output", str(source)]
    )
    with pytest.raises(SystemExit):
        main()
