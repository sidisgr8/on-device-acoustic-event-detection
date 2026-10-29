"""Confusion-matrix based error analysis from result files (protocol v2)."""
import numpy as np
import pandas as pd

from .aggregate import NAMES, load_all


def summed_confusion(folds) -> np.ndarray:
    return np.sum([np.array(r["confusion_matrix"]) for r in folds], axis=0)


def normalise_rows(cm: np.ndarray) -> np.ndarray:
    return cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)


def top_confusions(cm: np.ndarray, classes, n=5) -> pd.DataFrame:
    """Largest off-diagonal cells: true class mistaken for another class."""
    rows = []
    totals = cm.sum(axis=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                rows.append({"true": classes[i], "predicted": classes[j], "count": int(cm[i, j]),
                             "rate_of_true_class": float(cm[i, j] / max(totals[i], 1))})
    df = pd.DataFrame(rows).sort_values(["count", "rate_of_true_class"], ascending=False).head(n)
    return df.reset_index(drop=True)


def pooled_per_class_f1(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(axis=0), 1)
    rec = tp / np.maximum(cm.sum(axis=1), 1)
    return np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)


def top_confusions_table(results_dir="results", n=5) -> pd.DataFrame:
    """Per model, the n most frequent confusions (true class mistaken for another), summed over the folds shown."""
    rows = []
    for exp, a in analysis(results_dir).items():
        for rank, r in enumerate(a["top"].head(n).itertuples(), 1):
            rows.append({"Model": a["name"], "Folds": f"{len(a['folds'])}/10", "Rank": rank, "True class": r.true,
                         "Predicted as": r.predicted, "Clips": r.count,
                         "Share of true class (%)": f"{100 * r.rate_of_true_class:.1f}"})
    return pd.DataFrame(rows)


def analysis(results_dir="results") -> dict:
    """{experiment: {classes, cm, cm_norm, top, f1_mean_over_folds, f1_pooled, folds, acc_by_fold}}"""
    out = {}
    for exp, folds in load_all(results_dir).items():
        classes = folds[0]["classes"]
        cm = summed_confusion(folds)
        out[exp] = {"name": NAMES[exp], "classes": classes, "cm": cm, "cm_norm": normalise_rows(cm),
                    "top": top_confusions(cm, classes),
                    "f1_mean": np.array([np.mean([r["per_class_f1"][c] for r in folds]) for c in classes]),
                    "f1_pooled": pooled_per_class_f1(cm), "folds": [r["fold"] for r in folds],
                    "acc_by_fold": [r["accuracy"] for r in folds],
                    "n_params": folds[0]["n_params"]}
    return out
