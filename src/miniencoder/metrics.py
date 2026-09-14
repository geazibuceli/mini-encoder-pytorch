import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(expected, predicted):
    precision, recall, f1, _ = precision_recall_fscore_support(expected, predicted, average="binary", zero_division=0)
    return {"accuracy": float(accuracy_score(expected, predicted)), "precision": float(precision),
            "recall": float(recall), "f1": float(f1), "confusion_matrix": confusion_matrix(expected, predicted).tolist(),
            "num_examples": len(expected)}


def mean_confidence(logits):
    values = np.asarray(logits)
    values = np.exp(values - values.max(axis=1, keepdims=True))
    probabilities = values / values.sum(axis=1, keepdims=True)
    return float(probabilities.max(axis=1).mean())
