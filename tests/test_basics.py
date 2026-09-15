"""Basic safety tests (no dataset needed):  python -m pytest tests -q"""
import numpy as np
import torch

from src.data import N_FRAMES, N_MELS, fold_split, spec_augment, train_stats
from src.models import build_model


def fake_folds():
    return np.repeat(np.arange(1, 11), 20)            # 200 fake clips, 20 per fold


def test_split_has_no_leakage():
    fold = fake_folds()
    for k in range(1, 11):
        tr, va, te = fold_split(fold, k)
        assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
        assert (tr | va | te).all()
        assert set(fold[te]) == {k}
        assert set(fold[va]) == {k % 10 + 1}           # validation fold is never the test fold
        assert len(set(fold[tr])) == 8


def test_normalisation_uses_train_only():
    fold = fake_folds()
    X = np.zeros((200, 4, 4), np.float32)
    X[fold == 1] = 1000.0                               # test fold for k=1 is huge
    tr, _, _ = fold_split(fold, 1)
    mean, _ = train_stats(X, tr)
    assert mean == 0.0                                  # test-fold values must not leak into stats


def test_spec_augment_shape_and_input_untouched():
    x = np.ones((N_MELS, N_FRAMES), np.float32)
    y = spec_augment(x, np.random.default_rng(0))
    assert y.shape == x.shape and (x == 1).all()


def test_model_output_shapes():
    for name in ["cnn", "crnn", "mobilenet_scratch"]:
        m = build_model(name).eval()
        out = m(torch.randn(2, 1, N_MELS, N_FRAMES))
        assert out.shape == (2, 10), name
