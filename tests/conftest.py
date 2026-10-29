import numpy as np
import pandas as pd
import pytest

from src.data.dataset import FeatureStore
from src.features import params as P


@pytest.fixture
def synthetic_store():
    """10 folds x 12 clips; fold f has an offset of 10*f dB so any fold mix-up is visible."""
    rng = np.random.default_rng(0)
    rows = []
    for f in range(1, 11):
        for j in range(12):
            padded = j % 4 == 0
            nv = 60 if padded else P.N_FRAMES - 1
            rows.append(dict(fold=f, classID=j % P.N_CLASSES, n_valid_frames=nv, padded=padded,
                             slice_file_name=f"{f}-{j}.wav"))
    index = pd.DataFrame(rows)
    raw = np.empty((len(index), P.N_MELS, P.N_FRAMES), dtype=np.float32)
    for i, r in enumerate(index.itertuples()):
        raw[i] = -10.0 * r.fold + rng.normal(0, 1, (P.N_MELS, P.N_FRAMES))
        raw[i, :, r.n_valid_frames:] = -100.0  # padding, as in the real cache
    return FeatureStore(index=index, raw_logmel=raw)


@pytest.fixture(scope="session")
def real_cache():
    from pathlib import Path
    p = Path("data/cache")
    if not (p / "ast_fbank.npy").exists():
        pytest.skip("real cache not built (python scripts/build_cache.py)")
    return p


CLASSES = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
           "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]


def write_fold(root, exp, fold, acc=0.7, protocol=2, cfg_hash="h1", n_params=1000, seed=0):
    """Write a minimal synthetic result file of the shape produced by the training loop."""
    import json
    from pathlib import Path
    rng = np.random.default_rng(seed + fold)
    cm = rng.integers(0, 6, (10, 10))
    np.fill_diagonal(cm, rng.integers(30, 60, 10))
    r = {"experiment": exp, "fold": fold, "accuracy": acc, "macro_f1": acc - 0.02, "n_params": n_params,
         "classes": CLASSES, "per_class_f1": {c: float(0.5 + 0.04 * i) for i, c in enumerate(CLASSES)},
         "confusion_matrix": cm.tolist(), "config_hash": cfg_hash, "protocol_version": protocol,
         "by_padding": {"padded": {"n": 100, "accuracy": acc - 0.1}, "full": {"n": 700, "accuracy": acc}},
         "norm": {"mean": -24.0, "std": 17.0, "floor_db": -80.0}}
    d = Path(root) / exp
    d.mkdir(parents=True, exist_ok=True)
    (d / f"fold{fold}.json").write_text(json.dumps(r))
    return r
