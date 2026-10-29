import numpy as np
import torch

from src.data.dataset import compute_norm_stats, make_round, valid_frame_mask
from src.data.splits import round_indices
from src.features.logmel import apply_floor
from src.features import params as P

FLOOR = -80.0


def manual_stats(store, rows):
    vals = []
    raw = store.logmel(FLOOR)[:, 0].numpy()
    for i in rows:
        vals.append(raw[i][:, : store.n_valid[i]].ravel())
    v = np.concatenate(vals).astype(np.float64)
    return v.mean(), v.std()


def test_stats_use_train_folds_only(synthetic_store):
    for k in (1, 5, 10):
        tr, va, te = round_indices(synthetic_store.fold, k)
        _, _, _, st = make_round(synthetic_store, k, FLOOR)
        m, s = manual_stats(synthetic_store, tr)
        assert abs(st.mean - m) < 1e-3 and abs(st.std - s) < 1e-3
        all_m, _ = manual_stats(synthetic_store, np.arange(len(synthetic_store.fold)))
        # train-only differs from all-data stats because folds have different offsets
        assert abs(st.mean - all_m) > 1e-3


def test_changing_val_or_test_data_does_not_change_stats(synthetic_store):
    _, _, _, before = make_round(synthetic_store, 3, FLOOR)
    tr, va, te = round_indices(synthetic_store.fold, 3)
    synthetic_store._logmel[va] += 50.0
    synthetic_store._logmel[te] -= 50.0
    _, _, _, after = make_round(synthetic_store, 3, FLOOR)
    assert (before.mean, before.std) == (after.mean, after.std)


def test_padded_frames_excluded_from_stats(synthetic_store):
    tr, _, _ = round_indices(synthetic_store.fold, 2)
    x = synthetic_store.logmel(FLOOR)[tr]
    with_pad = compute_norm_stats(x, np.full(len(tr), P.N_FRAMES))   # pretend nothing is padding
    without = compute_norm_stats(x, synthetic_store.n_valid[tr])
    assert without.mean > with_pad.mean  # padding sits at the floor, pulling the mean down


def test_normalised_train_split_is_standardised_on_real_frames(synthetic_store):
    tr_ds, va_ds, te_ds, st = make_round(synthetic_store, 4, FLOOR)
    nv = synthetic_store.n_valid[tr_ds.rows.numpy()]
    m = valid_frame_mask(nv, P.N_FRAMES).bool().expand_as(tr_ds.x)
    v = tr_ds.x[m]
    assert abs(v.mean().item()) < 1e-3 and abs(v.std().item() - 1) < 1e-2


def test_floor_clamps_padding_not_content():
    x = torch.tensor([-100.0, -85.0, -80.0, -79.0, 10.0])
    assert apply_floor(x, -80.0).tolist() == [-80.0, -80.0, -80.0, -79.0, 10.0]
    assert apply_floor(x.numpy(), -80.0).min() == -80.0


def test_splits_are_built_from_the_right_folds(synthetic_store):
    tr, va, te, _ = make_round(synthetic_store, 10, FLOOR)
    f = synthetic_store.fold
    assert set(f[te.rows.numpy()]) == {10}
    assert set(f[va.rows.numpy()]) == {1}
    assert set(f[tr.rows.numpy()]) == set(range(2, 10))
