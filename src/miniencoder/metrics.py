import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(expected, predicted):
    if len(expected) == 0 or len(expected) != len(predicted):
        raise ValueError("Expected and predicted labels must be non-empty and have equal length.")
    precision, recall, f1, _ = precision_recall_fscore_support(
        expected, predicted, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=[0, 1]).tolist(),
        "num_examples": len(expected),
    }


def mean_confidence(logits):
    values = np.asarray(logits)
    if (
        values.ndim != 2
        or not values.shape[0]
        or values.shape[1] != 2
        or not np.isfinite(values).all()
    ):
        raise ValueError("Expected a non-empty finite matrix of binary classification logits.")
    values = np.exp(values - values.max(axis=1, keepdims=True))
    probabilities = values / values.sum(axis=1, keepdims=True)
    return float(probabilities.max(axis=1).mean())
