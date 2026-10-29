"""Fold arithmetic for the common protocol (interim spec, section 3).

Round k (1..10): test = fold k, validation = fold (k mod 10) + 1, train = the other 8 folds.
Folds are the dataset's predefined ones and are never re-split.
"""
import numpy as np

from src.features.params import N_FOLDS


def round_folds(k: int):
    """Return (train_folds, val_fold, test_fold) for round k, as (list[int], int, int)."""
    if not 1 <= k <= N_FOLDS:
        raise ValueError(f"round must be in 1..{N_FOLDS}, got {k}")
    test = k
    val = (k % N_FOLDS) + 1
    train = [f for f in range(1, N_FOLDS + 1) if f not in (test, val)]
    return train, val, test


def round_indices(fold_of_clip, k: int):
    """Row indices (train, val, test) into an array whose i-th entry is clip i's fold."""
    fold_of_clip = np.asarray(fold_of_clip)
    train, val, test = round_folds(k)
    return (np.flatnonzero(np.isin(fold_of_clip, train)),
            np.flatnonzero(fold_of_clip == val),
            np.flatnonzero(fold_of_clip == test))
