import pytest

from miniencoder.metrics import classification_metrics, mean_confidence


def test_metrics_known_predictions():
    report = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert report["accuracy"] == 0.75
    assert report["f1"] == pytest.approx(0.8)
    assert report["confusion_matrix"] == [[1, 1], [0, 2]]
    assert mean_confidence([[0, 0], [1000, 1000]]) == 0.5


def test_empty_metrics_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        classification_metrics([], [])
    with pytest.raises(ValueError, match="non-empty"):
        mean_confidence([])
