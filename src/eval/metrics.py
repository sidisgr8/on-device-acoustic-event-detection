import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.data.dataset import CLASSES

LABELS = list(range(len(CLASSES)))


def classification_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "per_class_f1": dict(zip(CLASSES, map(float, f1_score(
            y_true, y_pred, labels=LABELS, average=None, zero_division=0)))),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def subset_accuracy(y_true, y_pred, mask) -> dict:
    """Accuracy on clips selected by a boolean mask and on the rest (used for padded vs full)."""
    y_true, y_pred, mask = map(np.asarray, (y_true, y_pred, mask))
    out = {}
    for name, m in (("padded", mask), ("full", ~mask)):
        out[name] = {"n": int(m.sum()), "accuracy": float((y_true[m] == y_pred[m]).mean()) if m.any() else None}
    return out
