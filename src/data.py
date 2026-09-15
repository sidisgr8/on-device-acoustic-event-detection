"""Shared data pipeline: log-mel cache, fold splits, normalisation, SpecAugment.

Frozen spec (Interim report, Section 3):
  mono, 16 kHz, pad/crop to 4.0 s, n_fft 1024, hop 320, 128 mels, 20-8000 Hz,
  power-to-dB  ->  (128, 201) per clip.
Split for round k: test = fold k, val = fold (k mod 10) + 1, train = the other 8.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SR = 16000
DURATION = 4.0
N_SAMPLES = int(SR * DURATION)          # 64,000
N_FFT, HOP, N_MELS, FMIN, FMAX = 1024, 320, 128, 20, 8000
N_FRAMES = 1 + N_SAMPLES // HOP         # 201 (librosa centre padding)
CLASSES = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
           "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]


# ---------------------------------------------------------------- features
def log_mel(path):
    """One WAV file -> float32 log-mel spectrogram of shape (128, 201)."""
    import librosa                                       # imported here so tests need not have it
    y, _ = librosa.load(str(path), sr=SR, mono=True)     # load + resample to 16 kHz
    y = np.pad(y, (0, max(0, N_SAMPLES - len(y))))[:N_SAMPLES]   # zero-pad / crop to 4 s
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
    return librosa.power_to_db(m, ref=1.0, top_db=80.0).astype(np.float32)


def build_cache(root, cache_dir, n_jobs=-1):
    """Compute the log-mel of every clip once and store X.npy / y.npy / fold.npy."""
    from joblib import Parallel, delayed
    root, cache_dir = Path(root), Path(cache_dir)
    meta = pd.read_csv(root / "metadata" / "UrbanSound8K.csv")
    paths = [root / "audio" / f"fold{r.fold}" / r.slice_file_name for r in meta.itertuples()]
    X = Parallel(n_jobs=n_jobs, verbose=5)(delayed(log_mel)(p) for p in paths)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "X.npy", np.stack(X))
    np.save(cache_dir / "y.npy", meta["classID"].to_numpy(np.int64))
    np.save(cache_dir / "fold.npy", meta["fold"].to_numpy(np.int64))
    print(f"cached {len(X)} clips, shape {X[0].shape}")


def load_cache(cache_dir):
    cache_dir = Path(cache_dir)
    X = np.load(cache_dir / "X.npy", mmap_mode="r")
    return X, np.load(cache_dir / "y.npy"), np.load(cache_dir / "fold.npy")


# ------------------------------------------------------------------ splits
def fold_split(fold, k):
    """Boolean masks (train, val, test) for cross-validation round k (1..10)."""
    val_fold = k % 10 + 1
    test = fold == k
    val = fold == val_fold
    train = ~(test | val)
    return train, val, test


def train_stats(X, train_mask):
    """Mean/std of the TRAINING folds only (a single scalar pair)."""
    idx = np.where(train_mask)[0]
    sample = np.asarray(X[idx])
    return float(sample.mean()), float(sample.std() + 1e-8)


# ------------------------------------------------------------ augmentation
def spec_augment(x, rng, n_masks=2, max_freq=16, max_time=20):
    """SpecAugment-style masking on a (128, 201) array. Training batches only."""
    x = x.copy()
    for _ in range(n_masks):
        w = int(rng.integers(0, max_freq + 1)); f0 = int(rng.integers(0, x.shape[0] - w + 1))
        x[f0:f0 + w, :] = 0.0                    # frequency mask (0 = mean after normalisation)
        w = int(rng.integers(0, max_time + 1)); t0 = int(rng.integers(0, x.shape[1] - w + 1))
        x[:, t0:t0 + w] = 0.0                    # time mask
    return x


class ClipDataset(Dataset):
    """Cached log-mels for a set of indices, normalised, optionally augmented."""

    def __init__(self, X, y, indices, mean, std, augment=False, seed=42):
        self.X, self.y, self.idx = X, y, np.asarray(indices)
        self.mean, self.std, self.augment = mean, std, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = (np.asarray(self.X[j]) - self.mean) / self.std
        if self.augment:
            x = spec_augment(x, self.rng)
        return torch.from_numpy(x).unsqueeze(0), int(self.y[j])      # (1, 128, 201)
