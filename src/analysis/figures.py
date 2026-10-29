"""Figure set generated from result files only (reproducible: no timestamps, fixed palette).

Design rules (dataviz skill): one categorical colour slot per model in a fixed order (never cycled or
re-assigned when a model is missing); single-hue sequential ramp for magnitudes (confusions, F1);
recessive grid, thin marks, text in ink colours (never the series colour), direct labels, legend
whenever two series share a plot, no dual axes. Marker shape is a secondary encoding in scatter plots.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .aggregate import MODELS, NAMES, load_all
from .errors import analysis

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SLOT = {  # categorical slots 1-5 of the reference palette, fixed per model
    "cnn_baseline": "#2a78d6", "mobilenetv3_small_scratch": "#eb6834",
    "mobilenetv3_small_imagenet": "#1baf7a", "crnn": "#eda100", "ast": "#e87ba4",
}
MARKER = {"cnn_baseline": "o", "mobilenetv3_small_scratch": "s", "mobilenetv3_small_imagenet": "^",
          "crnn": "D", "ast": "P"}
SEQ = "Blues"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "regular", "axes.titlelocation": "left",
})


def _short(c):
    return c.replace("_", " ")


def _subtitle(a):
    n = len(a["folds"])
    return "" if n == 10 else f" [{n}/10 folds: {','.join(map(str, a['folds']))}]"


def fig_confusion(a, path):
    """Raw (left) and row-normalised (right) confusion matrix summed over folds."""
    cls = [_short(c) for c in a["classes"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, mat, title, fmt in ((axes[0], a["cm"], "Summed confusion matrix (clips)", "{:d}"),
                                (axes[1], a["cm_norm"], "Row-normalised (share of true class)", "{:.0%}")):
        im = ax.imshow(mat, cmap=SEQ, vmin=0, vmax=mat.max())
        ax.set_xticks(range(len(cls)), cls, rotation=45, ha="right")
        ax.set_yticks(range(len(cls)), cls)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.grid(False)
        thr = mat.max() * 0.55
        for i in range(len(cls)):
            for j in range(len(cls)):
                v = mat[i, j]
                if v > 0:
                    ax.text(j, i, fmt.format(int(v) if fmt == "{:d}" else v), ha="center", va="center",
                            fontsize=7, color="white" if v > thr else INK)
        ax.set_title(title)
    fig.suptitle(f"{a['name']}{_subtitle(a)}", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_confusion_grid(an, path):
    n = len(an)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4), squeeze=False)
    for ax, (exp, a) in zip(axes[0], an.items()):
        cls = [_short(c) for c in a["classes"]]
        ax.imshow(a["cm_norm"], cmap=SEQ, vmin=0, vmax=1)
        ax.set_xticks(range(len(cls)), cls, rotation=60, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(cls)), cls if ax is axes[0][0] else [""] * len(cls), fontsize=6.5)
        ax.grid(False)
        ax.set_title(a["name"] + _subtitle(a), fontsize=8.5)
    fig.suptitle("Row-normalised confusion matrices (true class on rows; summed over folds)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_top_confusions(an, path, n=5):
    k = len(an)
    fig, axes = plt.subplots(1, k, figsize=(4.4 * k, 3.2), squeeze=False, sharex=True)
    xmax = max((a["top"]["rate_of_true_class"].max() if len(a["top"]) else 0) for a in an.values()) * 100
    for ax, (exp, a) in zip(axes[0], an.items()):
        t = a["top"].head(n).iloc[::-1]
        labels = [f"{_short(r.true)} → {_short(r.predicted)}" for r in t.itertuples()]
        ax.barh(labels, t.rate_of_true_class * 100, color=SLOT[exp], height=0.6)
        for i, r in enumerate(t.itertuples()):
            ax.text(r.rate_of_true_class * 100 + xmax * 0.01, i, f"{r.rate_of_true_class:.0%} ({r.count})",
                    va="center", fontsize=7, color=INK)
        ax.set_title(a["name"] + _subtitle(a), fontsize=8.5)
        ax.set_xlim(0, xmax * 1.35)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("% of the true class (count)")
    fig.suptitle("Most frequent confusions (summed over folds)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_per_class_f1(an, path):
    names = [a["name"] for a in an.values()]
    cls = [_short(c) for c in next(iter(an.values()))["classes"]]
    mat = np.array([a["f1_mean"] for a in an.values()])
    fig, ax = plt.subplots(figsize=(1.0 * len(cls) + 2.5, 0.55 * len(names) + 1.6))
    im = ax.imshow(mat, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cls)), cls, rotation=40, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > 0.6 else INK)
    ax.set_title("Per-class F1 (mean over available folds)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_fold_accuracy(an, path):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for x, (exp, a) in enumerate(an.items()):
        acc = np.array(a["acc_by_fold"]) * 100
        ax.scatter(np.full(len(acc), x) + np.linspace(-0.12, 0.12, len(acc)), acc, s=18, color=SLOT[exp],
                   marker=MARKER[exp], edgecolor=SURFACE, linewidth=0.6, zorder=3)
        ax.hlines(acc.mean(), x - 0.28, x + 0.28, color=INK, lw=1.2, zorder=4)
        ax.text(x + 0.3, acc.mean(), f"{acc.mean():.1f}", va="center", fontsize=7.5)
    ax.set_xticks(range(len(an)), [a["name"] + ("\n" + f"{len(a['folds'])} folds" if len(a["folds"]) < 10 else "")
                                   for a in an.values()], fontsize=7.5)
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Accuracy per test fold (dots) and mean (bar)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_padded_vs_full(results_dir, path):
    from .aggregate import padded_vs_full_table
    df = padded_vs_full_table(results_dir)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(df))
    pa = df["Padded acc (%)"].astype(float).values
    fa = df["Full acc (%)"].astype(float).values
    ax.bar(x - 0.2, pa, 0.38, color="#52514e", label="padded clips (< 4 s)")
    ax.bar(x + 0.2, fa, 0.38, color="#2a78d6", label="full-length clips")
    for i in range(len(df)):
        ax.text(x[i] - 0.2, pa[i] + 0.5, f"{pa[i]:.1f}", ha="center", fontsize=7)
        ax.text(x[i] + 0.2, fa[i] + 0.5, f"{fa[i]:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x, df["Model"], fontsize=7.5)
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_title("Accuracy on padded vs full-length test clips (pooled over folds)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_accuracy_vs_cost(an, results_dir, path):
    """Accuracy against parameters (log axis), one marker per model, direct labels."""
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for exp, a in an.items():
        acc = np.array(a["acc_by_fold"]) * 100
        ax.errorbar(a["n_params"] / 1e6, acc.mean(), yerr=acc.std(ddof=1) if len(acc) > 1 else 0,
                    fmt=MARKER[exp], color=SLOT[exp], ms=7, mec=SURFACE, mew=0.8, capsize=2, lw=1)
        ax.annotate(a["name"], (a["n_params"] / 1e6, acc.mean()), textcoords="offset points", xytext=(7, 5),
                    fontsize=7.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("parameters (millions, log scale)")
    ax.set_ylabel("test accuracy (%), mean ± std over folds")
    ax.set_title("Accuracy vs model size")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _cost_scatter(an, x_of, xlabel, title, path, xscale="log"):
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for exp, a in an.items():
        x = x_of(exp, a)
        if x is None:
            continue
        acc = np.array(a["acc_by_fold"]) * 100
        ax.errorbar(x, acc.mean(), yerr=acc.std(ddof=1) if len(acc) > 1 else 0, fmt=MARKER[exp], color=SLOT[exp],
                    ms=7, mec=SURFACE, mew=0.8, capsize=2, lw=1)
        ax.annotate(a["name"] + ("" if len(a["folds"]) == 10 else f" ({len(a['folds'])}/10 folds)"), (x, acc.mean()),
                    textcoords="offset points", xytext=(7, 5), fontsize=7.5, color=INK)
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test accuracy (%), mean ± std over folds")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_accuracy_vs_macs(an, complexity, path):
    """Accuracy against multiply-accumulates per clip (log axis): the compute cost, which parameters do not show."""
    _cost_scatter(an, lambda e, a: complexity[e]["macs_total"] / 1e6 if e in complexity else None,
                  "multiply-accumulates per 4 s clip (millions, log scale)", "Accuracy vs compute (MACs)", path)


def fig_accuracy_vs_latency(an, latency_ms, path):
    """Accuracy against measured fp32 CPU latency (1 thread, median); only drawn once latency.json exists."""
    _cost_scatter(an, lambda e, a: latency_ms.get(e),
                  "CPU latency, batch 1, 1 thread, median (ms, log scale)", "Accuracy vs measured latency", path)


def fig_training_curves(results_dir, path):
    """Per model: train/validation loss (top) and accuracy (bottom) by epoch; line = mean over folds, band = min-max."""
    from .curves import load_curves, summarize
    exps = [e for e, _, _ in MODELS if load_curves(results_dir, e)]
    if not exps:
        return False
    fig, axes = plt.subplots(2, len(exps), figsize=(2.6 * len(exps), 5.0), squeeze=False)
    for j, exp in enumerate(exps):
        c = summarize(load_curves(results_dir, exp))
        ep = np.arange(1, len(c["train_loss"]["mean"]) + 1)
        for i, (lbl, kt, kv) in enumerate((("loss", "train_loss", "val_loss"), ("accuracy", "train_acc", "val_acc"))):
            ax = axes[i, j]
            for k, ls, name in ((kt, "-", "train"), (kv, "--", "validation")):
                ax.fill_between(ep, c[k]["lo"], c[k]["hi"], color=SLOT[exp], alpha=0.15, lw=0)
                ax.plot(ep, c[k]["mean"], ls, color=SLOT[exp], lw=1.4, label=name)
            if i == 0:
                ax.set_title(NAMES[exp], fontsize=8.5)
            if j == 0:
                ax.set_ylabel(lbl)
            if i == 1:
                ax.set_xlabel("epoch")
                ax.set_ylim(0, 1.0)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Training curves, mean over folds (band = min-max over folds). Train: augmented batches, label smoothing "
                 "0.1; validation: plain CE", x=0.01, ha="left", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)
    return True


def fig_paired_differences(results_dir, path, metric="accuracy"):
    """Forest plot of paired mean differences (A-B) with bootstrap 95 % CI; filled dot = Holm-adjusted Wilcoxon p < 0.05."""
    from .paired import paired_table
    df = paired_table(results_dir, [e for e, _, _ in MODELS], metric)
    if df.empty:
        return False
    scale = 100 if metric == "accuracy" else 1
    df = df.sort_values("_mean_diff").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.8, 0.38 * len(df) + 1.3))
    for y, r in df.iterrows():
        sig = r["_p_holm"] < 0.05
        ax.hlines(y, r["_ci_lo"] * scale, r["_ci_hi"] * scale, color=INK, lw=1.2)
        ax.plot(r["_mean_diff"] * scale, y, "o", ms=6, mfc=INK if sig else SURFACE, mec=INK, mew=1.1, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.9)
    ax.set_yticks(range(len(df)), [f"{r['A']}  -  {r['B']}" for _, r in df.iterrows()], fontsize=7.5)
    ax.set_xlabel("mean per-fold difference A - B (" + ("accuracy points" if metric == "accuracy" else "macro-F1") +
                  "), 95% bootstrap CI")
    ax.set_title("Paired differences (filled: Holm p < 0.05)", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def fig_accuracy_vs_size(an, results_dir, path, ckpt_dir="checkpoints"):
    """Accuracy against on-disk model size (log axis): fp32 models, plus int8 variants and pruned (gzip) models if present."""
    import json
    from matplotlib.lines import Line2D
    from .aggregate import _read_compress, fp32_size_mb
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for exp, a in an.items():
        size = fp32_size_mb(results_dir, ckpt_dir, exp)
        if size is None:
            continue
        acc = np.array(a["acc_by_fold"]) * 100
        ax.errorbar(size, acc.mean(), yerr=acc.std(ddof=1) if len(acc) > 1 else 0, fmt=MARKER[exp], color=SLOT[exp],
                    ms=7, mec=SURFACE, mew=0.8, capsize=2, lw=1)
        dx, dy, ha = {"cnn_baseline": (-9, -3, "right"), "crnn": (7, 4, "left")}.get(exp, (7, 5, "left"))
        ax.annotate(a["name"], (size, acc.mean()), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=7.5, color=INK)
    try:
        q, _ = _read_compress(results_dir, "quant")
        pr, _ = _read_compress(results_dir, "prune")
        pr = [r for r in pr if r.get("recipe", "main") == "main"]
    except FileNotFoundError:
        q, pr = [], []
    for exp in an:
        for v in sorted({r["variant"] for r in q if r["experiment"] == exp and "int8" in r["variant"]}):
            rs = [r for r in q if r["experiment"] == exp and r["variant"] == v and "accuracy" in r]
            if rs:
                ax.plot(np.mean([r["size_bytes"] for r in rs]) / 1e6, 100 * np.mean([r["accuracy"] for r in rs]),
                        MARKER[exp], ms=6, mfc=SURFACE, mec=SLOT[exp], mew=1.3)
        for sp in sorted({r["sparsity"] for r in pr if r["experiment"] == exp}):
            rs = [r for r in pr if r["experiment"] == exp and r["sparsity"] == sp]
            ax.plot(np.mean([r["size_gzip_bytes"] for r in rs]) / 1e6, 100 * np.mean([r["accuracy"] for r in rs]),
                    "x", ms=6, mec=SLOT[exp], mew=1.4)
    handles = [Line2D([], [], marker="o", ls="", mfc=MUTED, mec=MUTED, label="fp32 (filled; bars = std over folds)")]
    if q:
        handles.append(Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=MUTED, label="int8 quantized (hollow)"))
    if pr:
        handles.append(Line2D([], [], marker="x", ls="", mec=MUTED, label="pruned, gzip size (x; folds 1, 5, 9 only)"))
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="lower right")
    ax.set_xscale("log")
    ax.set_xlim(left=0.45)
    ax.set_xlabel("on-disk size (MB, log scale)")
    ax.set_ylabel("test accuracy (%), mean over folds")
    ax.set_title("Accuracy vs model size")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_all(results_dir="results", out_dir="results/figures") -> list:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    an = analysis(results_dir)
    if not an:
        return []
    written = []
    for exp, a in an.items():
        p = out / f"confusion_{exp}.png"
        fig_confusion(a, p)
        written.append(p)
    for name, fn in (("confusion_grid", lambda p: fig_confusion_grid(an, p)),
                     ("top_confusions", lambda p: fig_top_confusions(an, p)),
                     ("per_class_f1", lambda p: fig_per_class_f1(an, p)),
                     ("fold_accuracy", lambda p: fig_fold_accuracy(an, p)),
                     ("padded_vs_full", lambda p: fig_padded_vs_full(results_dir, p)),
                     ("accuracy_vs_params", lambda p: fig_accuracy_vs_cost(an, results_dir, p)),
                     ("accuracy_vs_size", lambda p: fig_accuracy_vs_size(an, results_dir, p)),
                     ("training_curves", lambda p: fig_training_curves(results_dir, p))):
        p = out / f"{name}.png"
        if fn(p) is not False:                                   # figures that need absent data return False
            written.append(p)
    for metric, name in (("accuracy", "paired_differences"), ("macro_f1", "paired_differences_macro_f1")):
        p = out / f"{name}.png"
        if len(an) > 1 and fig_paired_differences(results_dir, p, metric):
            written.append(p)
    from .aggregate import complexity_lookup, latency_lookup
    cx = complexity_lookup(results_dir)
    if any(e in cx for e in an):                                  # companion figures only when their data exists
        p = out / "accuracy_vs_macs.png"
        fig_accuracy_vs_macs(an, cx, p)
        written.append(p)
    lat = {k[0]: float(v) for k, v in latency_lookup(results_dir).items() if k[1] == "fp32_torch" and k[2] == 1}
    if any(e in lat for e in an):
        p = out / "accuracy_vs_latency.png"
        fig_accuracy_vs_latency(an, lat, p)
        written.append(p)
    return written
