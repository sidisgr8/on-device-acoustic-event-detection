import numpy as np
import pytest

from src.data.splits import round_folds, round_indices


@pytest.mark.parametrize("k", range(1, 11))
def test_fold_arithmetic_and_disjointness(k):
    train, val, test = round_folds(k)
    assert test == k
    assert val == (k % 10) + 1
    assert len(train) == 8
    assert set(train).isdisjoint({val, test}) and val != test
    assert sorted(train + [val, test]) == list(range(1, 11))


def test_explicit_table():
    assert round_folds(1) == ([3, 4, 5, 6, 7, 8, 9, 10], 2, 1)
    assert round_folds(9) == ([1, 2, 3, 4, 5, 6, 7, 8], 10, 9)
    assert round_folds(10) == ([2, 3, 4, 5, 6, 7, 8, 9], 1, 10)  # wrap-around: val of round 10 is fold 1


def test_invalid_round():
    for bad in (0, 11, -1):
        with pytest.raises(ValueError):
            round_folds(bad)


@pytest.mark.parametrize("k", range(1, 11))
def test_indices_partition_clips_without_leakage(k):
    rng = np.random.default_rng(k)
    fold_of_clip = rng.integers(1, 11, size=500)
    tr, va, te = round_indices(fold_of_clip, k)
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)
    assert len(tr) + len(va) + len(te) == 500
    assert set(fold_of_clip[te]) == {k}
    assert set(fold_of_clip[va]) == {(k % 10) + 1}
    assert k not in set(fold_of_clip[tr]) and (k % 10) + 1 not in set(fold_of_clip[tr])


def test_real_metadata_every_round_covers_all_clips(real_cache):
    import pandas as pd
    fold = pd.read_csv(real_cache / "index.csv").fold.values
    for k in range(1, 11):
        tr, va, te = round_indices(fold, k)
        assert len(np.unique(np.concatenate([tr, va, te]))) == len(fold)
