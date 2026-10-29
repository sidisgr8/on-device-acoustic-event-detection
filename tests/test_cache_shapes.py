import numpy as np
import pandas as pd
import pytest

from src.features import params as P
from src.features.logmel import ast_normalize, logmel_db, n_valid_frames


def test_n_valid_frames():
    assert n_valid_frames(P.N_SAMPLES) == 200
    assert n_valid_frames(1) == 1
    assert n_valid_frames(321) == 2 and n_valid_frames(320) == 1


def test_logmel_shape_and_silence_floor():
    y = np.zeros(P.N_SAMPLES, dtype=np.float32)
    S = logmel_db(y)
    assert S.shape == (P.N_MELS, P.N_FRAMES) == (128, 201)
    assert np.all(S == -100.0)


def test_cached_arrays_shapes_and_alignment(real_cache):
    idx = pd.read_csv(real_cache / "index.csv")
    meta = pd.read_csv("data/UrbanSound8K/metadata/UrbanSound8K.csv")
    assert len(idx) == 8732
    assert (idx.slice_file_name.values == meta.slice_file_name.values).all()   # row-aligned
    assert (idx.fold.values == meta.fold.values).all()
    wav = np.load(real_cache / "waveforms.npy", mmap_mode="r")
    lm = np.load(real_cache / "logmel_raw.npy", mmap_mode="r")
    ast = np.load(real_cache / "ast_fbank.npy", mmap_mode="r")
    assert wav.shape == (8732, 64000) and lm.shape == (8732, 128, 201) and ast.shape == (8732, 400, 128)
    assert wav.dtype == lm.dtype == ast.dtype == np.float32


def test_padding_bookkeeping_matches_waveforms(real_cache):
    idx = pd.read_csv(real_cache / "index.csv")
    wav = np.load(real_cache / "waveforms.npy", mmap_mode="r")
    lm = np.load(real_cache / "logmel_raw.npy", mmap_mode="r")
    assert idx.padded.sum() == 1399 and idx.cropped.sum() == 8     # as found in the EDA
    for i in np.flatnonzero(idx.padded.values)[:25]:
        n = idx.n_samples_16k[i]
        assert np.all(np.asarray(wav[i][n:]) == 0)
        nv = idx.n_valid_frames[i]
        assert np.all(np.asarray(lm[i][:, nv + 3:]) == -100.0)       # pure padding -> raw floor
    for i in np.flatnonzero(~idx.padded.values)[:25]:
        assert idx.n_valid_frames[i] == 200


def test_cached_logmel_finite_and_floor_effective(real_cache):
    lm = np.load(real_cache / "logmel_raw.npy", mmap_mode="r")
    sample = np.asarray(lm[::40])
    assert np.isfinite(sample).all() and sample.min() >= -100.0
    assert np.maximum(sample, -80.0).min() == -80.0


def test_cached_ast_matches_huggingface_feature_extractor(real_cache):
    from transformers import ASTFeatureExtractor
    wav = np.load(real_cache / "waveforms.npy", mmap_mode="r")
    ast = np.load(real_cache / "ast_fbank.npy", mmap_mode="r")
    fe = ASTFeatureExtractor(max_length=P.AST_FRAMES)
    for i in (0, 1234, 8000):
        ref = fe(np.asarray(wav[i]), sampling_rate=P.SR, return_tensors="np")["input_values"][0]
        assert ref.shape == (400, 128)
        np.testing.assert_allclose(ast_normalize(np.asarray(ast[i])), ref, atol=1e-4)
