"""Evidence for the log-mel dB-floor choice. Uses audio statistics only (no labels, no models).

Decision rule (logged in docs/DECISIONS.md, D-15): the floor is the lowest multiple of 10 dB
that clamps at most 1 % of the real (non-padded) time-frequency bins of the round-1 TRAINING
folds (folds 3-10; fold 1 is round-1 test and fold 2 its validation fold, so neither is
looked at). Per-fold percentiles of all 10 folds are reported only as a stability check.

    python scripts/db_floor_analysis.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.splits import round_folds

CACHE = Path("data/cache")
OUT = Path("results/eda")
CANDIDATES = [-100, -90, -80, -70, -60]
MAX_CLAMPED = 0.01


def valid_bins(X, nv, rows):
    return np.concatenate([np.asarray(X[i][:, : nv[i]]).ravel() for i in rows])


def main():
    X = np.load(CACHE / "logmel_raw.npy", mmap_mode="r")
    idx = pd.read_csv(CACHE / "index.csv")
    nv, fold = idx.n_valid_frames.values, idx.fold.values

    train_folds, _, _ = round_folds(1)
    rows = np.where(np.isin(fold, train_folds))[0]
    v = valid_bins(X, nv, rows)
    rows_pad = [i for i in rows if idx.padded[i]]
    pad = np.concatenate([np.asarray(X[i][:, nv[i] + 3:]).ravel() for i in rows_pad])

    table = {}
    for f in CANDIDATES:
        c = np.maximum(v, f)
        mean, std = float(c.mean()), float(c.std())
        table[f] = {"frac_valid_bins_clamped": float((v < f).mean()),
                    "valid_mean_db": mean, "valid_std_db": std,
                    "padding_in_sigma_units": (f - mean) / std}
    # Candidates are negative dB values; the floor closest to the real content (largest value)
    # that still clamps at most MAX_CLAMPED of real bins is chosen.
    ok = [f for f in CANDIDATES if table[f]["frac_valid_bins_clamped"] <= MAX_CLAMPED]
    chosen = max(ok)

    qs = [0.1, 1, 5, 25, 50, 75, 99]
    per_fold = {}
    for f in range(1, 11):
        vf = valid_bins(X, nv, np.where(fold == f)[0])
        per_fold[f] = {"percentiles_db": dict(zip(map(str, qs), map(float, np.percentile(vf, qs)))),
                       "frac_at_minus100": float((vf <= -99.99).mean()),
                       "frac_clamped_at_chosen": float((vf < chosen).mean())}
    mins = np.array([np.asarray(X[i][:, : nv[i]]).min() for i in range(len(idx))])
    result = {
        "rule": f"max floor (multiple of 10 dB) clamping <= {MAX_CLAMPED:.0%} of valid bins in folds {train_folds}",
        "analysis_folds": list(map(int, train_folds)),
        "candidates": table, "chosen_floor_db": chosen,
        "padding_raw_db": float(pad.min()), "padding_raw_is_constant": bool(pad.min() == pad.max()),
        "clips_with_digital_silence_in_valid_frames": int((mins <= -99.99).sum()),
        "per_fold_stability": per_fold,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "db_floor_analysis.json").write_text(json.dumps(result, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(v, bins=np.arange(-100, 45, 1.0), color="#2f6f9f", density=True)
    for f in CANDIDATES:
        ax[0].axvline(f, color="#6b7785" if f != chosen else "#c0392b", lw=1.2 if f == chosen else 0.8,
                      ls="-" if f == chosen else ":")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("log-mel bin value (dB re 1.0)")
    ax[0].set_ylabel("density (log)")
    ax[0].set_title(f"Real bins, folds {train_folds[0]}-{train_folds[-1]}; padding = -100 dB; floor {chosen} dB",
                    loc="left", fontsize=8.5)
    i = int(next(r for r in rows if idx.padded[r] and idx.class_name[r] == "gun_shot"))
    im = np.asarray(X[i])
    ax[1].imshow(np.maximum(im, chosen), origin="lower", aspect="auto", cmap="magma", vmin=-100, vmax=30,
                 extent=[0, 4, 0, 128])
    ax[1].axvline(nv[i] * 0.02, color="white", lw=0.8, ls="--")
    ax[1].set_title(f"{idx.slice_file_name[i]}: floored at {chosen} dB (dashed = end of real signal)",
                    loc="left", fontsize=8.5)
    ax[1].set_xlabel("time (s)")
    ax[1].grid(False)
    fig.tight_layout()
    fig.savefig(OUT / "db_floor_analysis.png", dpi=130)
    print(json.dumps({k: result[k] for k in ("rule", "candidates", "chosen_floor_db")}, indent=2))


if __name__ == "__main__":
    main()
