"""In-memory feature store, per-round normalisation, SpecAugment and batch Dataset."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler

from src.data.splits import round_indices
from src.features import params as P
from src.features.logmel import apply_floor

CLASSES = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
           "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]  # classID order


@dataclass
class NormStats:
    mean: float
    std: float


def valid_frame_mask(n_valid_frames, n_frames: int) -> torch.Tensor:
    """(n, 1, 1, T) float mask, 1 for frames centred on real (unpadded) signal."""
    nv = torch.as_tensor(np.asarray(n_valid_frames))
    return (torch.arange(n_frames)[None, :] < nv[:, None]).float()[:, None, None, :]


def compute_norm_stats(x: torch.Tensor, n_valid_frames, chunk: int = 512) -> NormStats:
    """Scalar mean/std of a (n, 1, F, T) log-mel tensor over REAL frames only.

    Padded frames are excluded so the statistics do not depend on how much padding the
    training clips happen to contain (padding is class-correlated, see docs/DECISIONS.md D-15).
    Accumulated in float64.
    """
    total = count = sq = 0.0
    for s in range(0, len(x), chunk):
        xc = x[s:s + chunk].double()
        m = valid_frame_mask(n_valid_frames[s:s + chunk], x.shape[-1]).double()
        total += (xc * m).sum().item()
        sq += (xc * xc * m).sum().item()
        count += m.sum().item() * x.shape[-2]
    mean = total / count
    return NormStats(mean=mean, std=float(np.sqrt(max(sq / count - mean * mean, 1e-12))))


class SpecAugment(torch.nn.Module):
    """Frequency and time masking on (B, C, F, T) tensors. Identity unless `.training`.

    Masked regions are set to 0, which is the dataset mean once inputs are normalised.
    Randomness comes from `self.generator` (a CPU torch.Generator) so runs are reproducible
    and resumable independently of device RNG state.
    """

    def __init__(self, freq_masks=2, freq_width=16, time_masks=2, time_width=20, generator=None,
                 freq_dim=2, time_dim=3):
        """Defaults suit (B, 1, F, T); AST inputs are (B, T, F): freq_dim=2, time_dim=1."""
        super().__init__()
        self.freq_dim, self.time_dim = freq_dim, time_dim
        self.freq_masks, self.freq_width = freq_masks, freq_width
        self.time_masks, self.time_width = time_masks, time_width
        self.generator = generator

    def _mask(self, x, dim, n_masks, max_width):
        B, L = x.shape[0], x.shape[dim]
        pos = torch.arange(L)[None, :]
        shape = [B] + [1] * (x.dim() - 1)
        shape[dim] = L
        for _ in range(n_masks):
            w = torch.randint(0, max_width + 1, (B,), generator=self.generator)
            start = (torch.rand(B, generator=self.generator) * (L - w + 1).float()).long()
            m = (pos >= start[:, None]) & (pos < (start + w)[:, None])
            x = x.masked_fill(m.view(shape).to(x.device), 0.0)
        return x

    def forward(self, x):
        if not self.training:
            return x
        x = self._mask(x, self.freq_dim, self.freq_masks, self.freq_width)
        return self._mask(x, self.time_dim, self.time_masks, self.time_width)


class FeatureStore:
    """Loads the index and the raw log-mel cache once; hands out per-round tensors."""

    def __init__(self, cache="data/cache", index: pd.DataFrame | None = None,
                 raw_logmel: np.ndarray | None = None):
        """Normally reads `cache`; tests pass `index` and `raw_logmel` directly."""
        self.cache = Path(cache)
        self.index = pd.read_csv(self.cache / "index.csv") if index is None else index
        self.fold = self.index.fold.values
        self.labels = torch.as_tensor(self.index.classID.to_numpy(copy=True), dtype=torch.long)
        self.n_valid = self.index.n_valid_frames.values
        self.padded = self.index.padded.values
        self._logmel = None if raw_logmel is None else torch.as_tensor(raw_logmel).unsqueeze(1)
        self._ast = None

    def logmel(self, floor_db: float) -> torch.Tensor:
        """(N, 1, 128, 201) float32 with the dB floor applied. Not normalised."""
        if self._logmel is None:
            raw = np.load(self.cache / "logmel_raw.npy")
            self._logmel = torch.from_numpy(raw).unsqueeze(1)
        return apply_floor(self._logmel, floor_db)


    def ast(self) -> torch.Tensor:
        """(N, 400, 128) AST fbank normalised with the checkpoint constants (x - mean) / (2 std)."""
        if self._ast is None:
            raw = torch.from_numpy(np.load(self.cache / "ast_fbank.npy"))
            self._ast = (raw - P.AST_MEAN) / (P.AST_STD * 2)
        return self._ast


class BatchDataset(Dataset):
    """Holds normalised tensors for one split. `ds[list_of_positions]` returns a whole batch,
    so it plugs into DataLoader(batch_size=None, sampler=BatchSampler(...)) without a collate
    cost. A scalar index returns a single item."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, rows: np.ndarray):
        self.x, self.y, self.rows = x, y, torch.as_tensor(rows)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.x[i], self.y[i], self.rows[i]
        i = torch.as_tensor(i)
        return self.x[i], self.y[i], self.rows[i]


def make_round(store: FeatureStore, k: int, floor_db: float):
    """Build (train, val, test) BatchDatasets and the NormStats for round k.

    Normalisation statistics come from the training folds ONLY.
    """
    tr, va, te = round_indices(store.fold, k)
    x_all = store.logmel(floor_db)
    stats = compute_norm_stats(x_all[tr], store.n_valid[tr])

    def split(rows):
        x = (x_all[rows] - stats.mean) / stats.std
        return BatchDataset(x, store.labels[rows], rows)

    return split(tr), split(va), split(te), stats


def make_round_ast(store: FeatureStore, k: int):
    """AST variant of `make_round`: features are normalised with the checkpoint's own constants
    (no statistics are estimated from any fold). Returns (train, val, test, NormStats)."""
    tr, va, te = round_indices(store.fold, k)
    x_all = store.ast()
    stats = NormStats(mean=P.AST_MEAN, std=2 * P.AST_STD)
    return (*(BatchDataset(x_all[r], store.labels[r], r) for r in (tr, va, te)), stats)


def train_loader(ds: BatchDataset, batch_size: int, seed: int, epoch: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed * 100_003 + epoch)
    sampler = BatchSampler(RandomSampler(ds, generator=g), batch_size, drop_last=True)
    return DataLoader(ds, sampler=sampler, batch_size=None)


def eval_loader(ds: BatchDataset, batch_size: int) -> DataLoader:
    sampler = BatchSampler(range(len(ds)), batch_size, drop_last=False)
    return DataLoader(ds, sampler=sampler, batch_size=None)
