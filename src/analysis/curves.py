"""Training curves per model, averaged over folds (mean line, min-max band), from the per-epoch logs.

Source: results/<exp>/fold<k>_epochs.jsonl. Caveat shown in the figure: training loss/accuracy are measured on
augmented (SpecAugment) batches with label smoothing 0.1 while the model is in training mode; validation loss
is plain cross-entropy on un-augmented clips, so the two loss levels are not directly comparable.
"""
import json
from pathlib import Path

import numpy as np

from .aggregate import NAMES


def load_curves(results_dir, exp) -> dict:
    """{metric: array (n_folds, n_epochs)} for train_loss, val_loss, train_acc, val_acc; folds with equal epoch counts."""
    runs = []
    for p in sorted(Path(results_dir, exp).glob("fold*_epochs.jsonl")):
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if rows:
            runs.append(rows)
    if not runs:
        return {}
    n = max(len(r) for r in runs)
    runs = [r for r in runs if len(r) == n]            # drop interrupted logs
    return {k: np.array([[row[k] for row in r] for r in runs]) for k in ("train_loss", "val_loss", "train_acc", "val_acc")}


def summarize(curves: dict) -> dict:
    return {k: {"mean": v.mean(0), "lo": v.min(0), "hi": v.max(0)} for k, v in curves.items()}
