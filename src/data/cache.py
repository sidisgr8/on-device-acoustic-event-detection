"""Build the shared on-disk caches from the raw UrbanSound8K audio.

Cache layout (data/cache/, gitignored), all arrays row-aligned with `index.csv`, which has
the same row order as metadata/UrbanSound8K.csv:

    index.csv          per-clip bookkeeping (original length, padded/cropped flags, ...)
    waveforms.npy      float32 (N, 64000)   16 kHz mono, zero-padded / cropped to 4.0 s
    logmel_raw.npy     float32 (N, 128, 201) power dB re 1.0, NO floor clamp
    ast_fbank.npy      float32 (N, 400, 128) un-normalised AST fbank
    cache_meta.json    parameters and library versions used
"""
import json
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from src.features import params as P
from src.features.logmel import ast_fbank, logmel_db, n_valid_frames

DEFAULT_RAW = Path("data/UrbanSound8K")
DEFAULT_CACHE = Path("data/cache")


def _load_one(args):
    path, = args
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    channels = y.shape[1]
    mono = y.mean(axis=1)
    if sr != P.SR:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=P.SR)  # soxr_hq
    n = len(mono)
    cropped = n > P.N_SAMPLES
    out = np.zeros(P.N_SAMPLES, dtype=np.float32)
    k = min(n, P.N_SAMPLES)
    out[:k] = mono[:k]
    return out, dict(orig_sr=sr, orig_channels=channels, orig_samples=y.shape[0],
                     n_samples_16k=k, cropped=bool(cropped), peak=float(np.abs(out).max()))


def build_waveforms(raw_root=DEFAULT_RAW, cache=DEFAULT_CACHE, workers=8):
    raw_root, cache = Path(raw_root), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(raw_root / "metadata" / "UrbanSound8K.csv")
    paths = [(str(raw_root / "audio" / f"fold{r.fold}" / r.slice_file_name),) for r in meta.itertuples()]
    wav = np.lib.format.open_memmap(cache / "waveforms.npy", mode="w+", dtype=np.float32,
                                    shape=(len(meta), P.N_SAMPLES))
    rows = []
    with Pool(workers) as pool:
        for i, (y, info) in enumerate(pool.imap(_load_one, paths, chunksize=16)):
            wav[i] = y
            rows.append(info)
            if (i + 1) % 1000 == 0:
                print(f"  waveforms {i + 1}/{len(meta)}", flush=True)
    wav.flush()
    info = pd.DataFrame(rows)
    idx = meta.rename(columns={"class": "class_name"}).copy()
    idx = pd.concat([idx, info], axis=1)
    idx["orig_duration_s"] = idx.orig_samples / idx.orig_sr
    idx["n_valid_frames"] = [n_valid_frames(n) for n in idx.n_samples_16k]
    idx["padded"] = idx.n_samples_16k < P.N_SAMPLES
    idx.to_csv(cache / "index.csv", index=False)
    return idx


def _logmel_one(y):
    return logmel_db(y)


def _fbank_one(y):
    return ast_fbank(y)


def _build_features(cache, name, fn, shape_tail, workers):
    cache = Path(cache)
    wav = np.load(cache / "waveforms.npy", mmap_mode="r")
    out = np.lib.format.open_memmap(cache / name, mode="w+", dtype=np.float32,
                                    shape=(len(wav), *shape_tail))
    with Pool(workers) as pool:
        for i, f in enumerate(pool.imap(fn, (np.asarray(w) for w in wav), chunksize=16)):
            out[i] = f
            if (i + 1) % 2000 == 0:
                print(f"  {name} {i + 1}/{len(wav)}", flush=True)
    out.flush()


def build_logmel(cache=DEFAULT_CACHE, workers=8):
    _build_features(cache, "logmel_raw.npy", _logmel_one, (P.N_MELS, P.N_FRAMES), workers)


def build_ast(cache=DEFAULT_CACHE, workers=4):
    _build_features(cache, "ast_fbank.npy", _fbank_one, (P.AST_FRAMES, P.AST_MEL_BINS), workers)


def write_meta(cache=DEFAULT_CACHE):
    import librosa, torch, torchaudio
    meta = {
        "sr": P.SR, "n_samples": P.N_SAMPLES, "n_fft": P.N_FFT, "hop": P.HOP, "n_mels": P.N_MELS,
        "fmin": P.FMIN, "fmax": P.FMAX, "amin": P.AMIN, "resampler": "librosa soxr_hq",
        "downmix": "mean over channels", "pad": "zeros at end", "crop": "first 4.0 s",
        "ast": {"frames": P.AST_FRAMES, "mel_bins": P.AST_MEL_BINS, "mean": P.AST_MEAN, "std": P.AST_STD},
        "versions": {"librosa": librosa.__version__, "torch": torch.__version__,
                     "torchaudio": torchaudio.__version__, "numpy": np.__version__},
    }
    (Path(cache) / "cache_meta.json").write_text(json.dumps(meta, indent=2))
