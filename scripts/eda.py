"""Exploratory analysis of UrbanSound8K: integrity checks, statistics, figures.

    python scripts/eda.py [--root data/UrbanSound8K] [--out results/eda]

Outputs (all small, tracked in git):
    eda_stats.json        headline numbers and the check against the interim claims
    class_counts.csv      clips per class
    fold_class_counts.csv clips per (fold, class)
    audio_info.csv        per-clip header info (sample rate, channels, duration, ...) and peak level
    *.png                 figures
"""
import argparse
import json
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

# Claims made in the interim report, section 3 (to be checked, not trusted).
INTERIM_CLAIMS = {"gun_shot": 374, "car_horn": 429, "siren": 929}
INTERIM_REST = 1000

INK, MUTED, GRID = "#1f2933", "#6b7785", "#e4e7eb"
BAR = "#2f6f9f"
plt.rcParams.update({
    "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 9,
})

# Front-end parameters from the interim spec, used here only to *display* spectrograms.
SR, N_FFT, HOP, N_MELS, FMIN, FMAX = 16000, 1024, 320, 128, 20, 8000


def clip_path(root, row):
    return root / "audio" / f"fold{row.fold}" / row.slice_file_name


def collect_audio_info(root, meta):
    rows = []
    for r in meta.to_dict("records"):
        p = root / "audio" / f"fold{r['fold']}" / r["slice_file_name"]
        info = sf.info(p)
        y, _ = sf.read(p, dtype="float32", always_2d=True)
        mono = y.mean(axis=1)
        rows.append({
            "slice_file_name": r["slice_file_name"], "fold": r["fold"], "class": r["class"],
            "sr": info.samplerate, "channels": info.channels, "frames": info.frames,
            "duration_s": info.frames / info.samplerate, "subtype": info.subtype,
            "peak": float(np.abs(mono).max()), "rms": float(np.sqrt((mono ** 2).mean())),
        })
    return pd.DataFrame(rows)


def fig_class_counts(counts, path):
    counts = counts.sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh(counts.index, counts.values, color=BAR, height=0.7)
    for i, v in enumerate(counts.values):
        ax.text(v + 12, i, str(v), va="center", color=INK)
    ax.set_xlabel("clips")
    ax.set_title("UrbanSound8K clips per class", loc="left", color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, counts.max() * 1.1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_fold_class(table, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.imshow(table.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=40, ha="right")
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.set_ylabel("fold")
    thresh = table.values.max() * 0.6
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            ax.text(j, i, v, ha="center", va="center", fontsize=7.5,
                    color="white" if v > thresh else INK)
    ax.grid(False)
    ax.set_title("Clips per fold and class", loc="left", color=INK)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_duration(info, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    ax.hist(info.duration_s, bins=np.arange(0, 4.21, 0.1), color=BAR)
    ax.set_xlabel("duration (s)")
    ax.set_ylabel("clips")
    ax.set_yscale("log")
    ax.set_title("Clip duration (log count)", loc="left", color=INK)
    ax = axes[1]
    sr_counts = info.sr.value_counts().sort_index()
    ax.bar([f"{s / 1000:.3g}" for s in sr_counts.index], sr_counts.values, color=BAR)
    for i, v in enumerate(sr_counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_xlabel("native sample rate (kHz)")
    ax.set_ylabel("clips")
    ax.set_title("Sample-rate variety", loc="left", color=INK)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_examples(root, meta, path, seed=0):
    classes = sorted(meta["class"].unique())
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(classes), 2, figsize=(9, 1.45 * len(classes)))
    for i, c in enumerate(classes):
        sub = meta[meta["class"] == c]
        r = sub.iloc[rng.integers(len(sub))]
        y, _ = librosa.load(clip_path(root, r), sr=SR, mono=True)
        y = np.pad(y, (0, max(0, 4 * SR - len(y))))[: 4 * SR]
        t = np.arange(len(y)) / SR
        axes[i, 0].plot(t, y, color=BAR, lw=0.4)
        axes[i, 0].set_ylim(-1, 1)
        axes[i, 0].set_ylabel(c, rotation=0, ha="right", va="center", fontsize=8)
        S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
                                           fmin=FMIN, fmax=FMAX, power=2.0)
        axes[i, 1].imshow(librosa.power_to_db(S, ref=1.0, top_db=None).clip(-80, None),
                          origin="lower", aspect="auto", cmap="magma", extent=[0, 4, 0, N_MELS])
        for ax in axes[i]:
            ax.grid(False)
            if i < len(classes) - 1:
                ax.set_xticklabels([])
        axes[i, 1].set_yticks([])
    axes[0, 0].set_title("waveform (16 kHz, padded to 4 s)", loc="left", fontsize=9)
    axes[0, 1].set_title("log-mel (128 mels, 20-8000 Hz)", loc="left", fontsize=9)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/UrbanSound8K")
    ap.add_argument("--out", default="results/eda")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(root / "metadata" / "UrbanSound8K.csv")
    stats = {"n_clips_metadata": int(len(meta)), "n_folds": int(meta.fold.nunique()),
             "n_classes": int(meta["class"].nunique())}

    missing = [r.slice_file_name for r in meta.itertuples() if not clip_path(root, r).exists()]
    n_wav = len(list((root / "audio").glob("fold*/*.wav")))
    stats.update(n_wav_on_disk=n_wav, n_metadata_files_missing=len(missing))
    assert not missing, f"{len(missing)} files listed in metadata are missing, e.g. {missing[:3]}"

    # Class and fold statistics
    counts = meta["class"].value_counts()
    counts.rename_axis("class").rename("clips").to_csv(out / "class_counts.csv")
    ft = pd.crosstab(meta.fold, meta["class"])
    ft.to_csv(out / "fold_class_counts.csv")
    stats["class_counts"] = counts.sort_index().to_dict()
    stats["clips_per_fold"] = {int(k): int(v) for k, v in meta.fold.value_counts().sort_index().items()}

    claims = {}
    for c, n in counts.items():
        expected = INTERIM_CLAIMS.get(c, INTERIM_REST)
        claims[c] = {"actual": int(n), "interim_claim": expected, "matches": int(n) == expected}
    stats["interim_claim_check"] = claims

    # Source-recording leakage check: a fsID must live in exactly one fold
    folds_per_source = meta.groupby("fsID").fold.nunique()
    stats["n_source_recordings"] = int(meta.fsID.nunique())
    stats["sources_spanning_multiple_folds"] = int((folds_per_source > 1).sum())
    stats["salience_counts"] = {int(k): int(v) for k, v in meta.salience.value_counts().sort_index().items()}

    # Audio header info + level
    info = collect_audio_info(root, meta)
    info.to_csv(out / "audio_info.csv", index=False)
    d = info.duration_s
    at4 = (info.frames >= 4 * info.sr - 1)  # full 4 s (allow 1 sample rounding)
    stats["duration"] = {
        "min": round(float(d.min()), 4), "median": round(float(d.median()), 4),
        "mean": round(float(d.mean()), 4), "max": round(float(d.max()), 4),
        "n_full_4s": int(at4.sum()), "n_shorter_than_4s": int((~at4).sum()),
        "n_longer_than_4s": int((info.frames > 4 * info.sr + 1).sum()),
        "n_shorter_than_1s": int((d < 1).sum()), "n_shorter_than_2s": int((d < 2).sum()),
    }
    stats["short_clips_by_class"] = info[~at4]["class"].value_counts().sort_index().to_dict()
    stats["sample_rates"] = {int(k): int(v) for k, v in info.sr.value_counts().sort_index().items()}
    stats["channels"] = {int(k): int(v) for k, v in info.channels.value_counts().sort_index().items()}
    stats["subtypes"] = info.subtype.value_counts().to_dict()
    stats["n_all_zero_clips"] = int((info.peak == 0).sum())
    stats["n_peak_below_0.001"] = int((info.peak < 1e-3).sum())

    fig_class_counts(counts, out / "class_counts.png")
    fig_fold_class(ft, out / "fold_class_counts.png")
    fig_duration(info, out / "duration_samplerate.png")
    fig_examples(root, meta, out / "examples_waveform_logmel.png")

    with open(out / "eda_stats.json", "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
