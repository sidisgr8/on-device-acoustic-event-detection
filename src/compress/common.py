"""Shared helpers for post-training tools: locate configs/checkpoints, rebuild trained models, build
evaluation and calibration data. Read-only with respect to checkpoints."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.dataset import FeatureStore, make_round, make_round_ast
from src.eval.metrics import classification_metrics
from src.models import build_model

CKPT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")
CALIB_CLIPS = 100
CALIB_SEED = 42


def find_config(experiment: str, config_dir="configs") -> dict:
    for p in sorted(Path(config_dir).glob("*.yaml")):
        cfg = yaml.safe_load(open(p))
        if isinstance(cfg, dict) and cfg.get("experiment") == experiment:
            return cfg
    raise FileNotFoundError(f"no config with experiment: {experiment}")


def model_kind(experiment_cfg: dict) -> str:
    return experiment_cfg["model"]["name"]  # cnn_baseline | mobilenetv3_small | crnn | ast


def best_ckpt(experiment: str, fold: int, ckpt_dir=CKPT_DIR) -> Path:
    return Path(ckpt_dir) / experiment / f"fold{fold}" / "best.pt"


def available_folds(experiment: str, ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR):
    """Folds that are FINISHED (v2 result exists) and have a best.pt."""
    out = []
    for k in range(1, 11):
        rj = Path(results_dir) / experiment / f"fold{k}.json"
        if rj.exists() and best_ckpt(experiment, k, ckpt_dir).exists():
            if json.loads(rj.read_text()).get("protocol_version") == 2:
                out.append(k)
    return out


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def load_trained_model(cfg: dict, ckpt_path, pretrained_init=False):
    """Rebuild the architecture on CPU (no pretrained download needed) and load best.pt."""
    mcfg = dict(cfg["model"])
    if "pretrained" in mcfg and not pretrained_init:
        mcfg["pretrained"] = False  # weights come from best.pt; avoid touching the network
    model = build_model(mcfg)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.eval()


def round_data(store: FeatureStore, cfg: dict, fold: int, split: str, n_calib=CALIB_CLIPS):
    """Return (eval_dataset, calibration_tensor, rows_of_calibration). `split` is 'val' or 'test'.

    Normalisation is the round's own (training-fold statistics, or AST constants). Calibration clips
    are drawn from the TRAINING folds only, with a fixed seed."""
    dcfg = cfg["data"]
    if dcfg.get("input", "logmel") == "ast":
        tr, va, te, stats = make_round_ast(store, fold)
    else:
        tr, va, te, stats = make_round(store, fold, dcfg["floor_db"])
    ds = {"val": va, "test": te}[split]
    rng = np.random.default_rng(CALIB_SEED)
    pick = np.sort(rng.permutation(len(tr))[:n_calib])
    return ds, tr.x[pick].clone(), tr.rows[pick].numpy(), stats


def evaluate_runner(runner, ds, batch_size=32, limit=None):
    """Run `runner(x)` over a dataset on CPU; return metrics + n."""
    n = len(ds) if limit is None else min(limit, len(ds))
    preds = []
    for s in range(0, n, batch_size):
        x = ds.x[s:min(s + batch_size, n)]
        with torch.no_grad():
            preds.append(runner(x).argmax(1).cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = ds.y[:n].numpy()
    m = classification_metrics(y_true, y_pred)
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "n_eval": int(n),
            "y_pred": y_pred, "y_true": y_true}
