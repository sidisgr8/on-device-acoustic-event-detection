"""Paired per-fold comparisons between models (reporting analysis, not tuning).

All models are evaluated on the same 10 predefined test folds, so the per-fold differences are paired.
For every model pair (A, B) and metric: mean difference A-B over folds, number of folds A is ahead,
Wilcoxon signed-rank test (exact, two-sided), paired t-test (df = folds-1, for reference), and a percentile
bootstrap 95 % CI of the mean difference obtained by resampling FOLDS with replacement (fixed seed).
Holm-adjusted Wilcoxon p-values are given within each metric (family = all pairs listed together).

Limits (stated in the table notes): 10 folds; one seed per model; the training sets of different folds overlap
(each pair of folds shares 7 of 8 training folds), so fold differences are not independent draws and every
p-value / interval here is optimistic about generalisation beyond this dataset's folds; with n=10 the smallest
possible two-sided exact Wilcoxon p is 2/1024 = 0.00195.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from .aggregate import NAMES, load_all

BOOT_N = 10000
BOOT_SEED = 0


def fold_metric(data: dict, exp: str, metric: str) -> dict:
    return {r["fold"]: r[metric] for r in data[exp]}


def paired_stats(a: np.ndarray, b: np.ndarray, boot_n=BOOT_N, seed=BOOT_SEED) -> dict:
    """a, b: per-fold metric values over the same folds (same order)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    out = {"n": n, "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)) if n > 1 else float("nan"),
           "a_wins": int((d > 0).sum()), "ties": int((d == 0).sum())}
    nz = d[d != 0]
    if len(nz) >= 1:
        out["wilcoxon_p"] = float(stats.wilcoxon(nz, method="exact" if len(nz) <= 50 else "approx",
                                                 alternative="two-sided").pvalue)
    else:
        out["wilcoxon_p"] = 1.0
    if n < 2:
        out["t_p"] = float("nan")
    elif d.std(ddof=1) == 0:                                  # constant difference: t is 0 or infinite
        out["t_p"] = 1.0 if d.mean() == 0 else 0.0
    else:
        out["t_p"] = float(stats.ttest_rel(a, b).pvalue)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(boot_n, n))
    means = d[idx].mean(axis=1)
    out["ci_lo"], out["ci_hi"] = (float(x) for x in np.percentile(means, [2.5, 97.5]))
    return out


def holm(p: list) -> list:
    """Holm-Bonferroni adjusted p-values (step-down), same order as the input."""
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def paired_records(results_dir="results", experiments=None, metric="accuracy", data=None) -> list:
    """One record per pair of experiments that share all of their folds (only the common folds are used)."""
    data = data if data is not None else load_all(results_dir)
    exps = [e for e in (experiments or list(data)) if e in data]
    recs = []
    for ea, eb in combinations(exps, 2):
        ma, mb = fold_metric(data, ea, metric), fold_metric(data, eb, metric)
        folds = sorted(set(ma) & set(mb))
        if len(folds) < 2:
            continue
        s = paired_stats(np.array([ma[k] for k in folds]), np.array([mb[k] for k in folds]))
        recs.append({"a": ea, "b": eb, "folds": folds, **s})
    if recs:
        for r, adj in zip(recs, holm([r["wilcoxon_p"] for r in recs])):
            r["wilcoxon_p_holm"] = adj
    return recs


def _fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def paired_table(results_dir="results", experiments=None, metric="accuracy", data=None) -> pd.DataFrame:
    scale, unit = (100.0, "pp") if metric == "accuracy" else (1.0, "")
    rows = []
    for r in paired_records(results_dir, experiments, metric, data):
        nf = len(r["folds"])
        rows.append({
            "A": NAMES.get(r["a"], r["a"]), "B": NAMES.get(r["b"], r["b"]), "Folds": f"{nf}/10",
            f"Mean diff A-B ({unit or 'F1'})": f"{r['mean_diff'] * scale:+.2f}" if unit else f"{r['mean_diff']:+.3f}",
            "95% bootstrap CI": (f"[{r['ci_lo'] * scale:+.2f}, {r['ci_hi'] * scale:+.2f}]" if unit
                                 else f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"),
            "A ahead in": f"{r['a_wins']}/{nf}" + (f" ({r['ties']} tied)" if r["ties"] else ""),
            "Wilcoxon p": _fmt_p(r["wilcoxon_p"]), "Wilcoxon p (Holm)": _fmt_p(r["wilcoxon_p_holm"]),
            "paired t p": _fmt_p(r["t_p"]),
            "_a": r["a"], "_b": r["b"], "_mean_diff": r["mean_diff"], "_ci_lo": r["ci_lo"], "_ci_hi": r["ci_hi"],
            "_p_holm": r["wilcoxon_p_holm"],
        })
    return pd.DataFrame(rows)


PAIRED_NOTES = [
    "Same 10 predefined test folds for every model, so differences are paired by fold; A-B > 0 means A is better. "
    "CI = percentile bootstrap (10,000 resamples of the folds, seed 0) of the mean difference.",
    "Limits: 10 folds and one training seed per model; training sets of different folds overlap, so folds are not "
    "independent samples and these p-values and intervals are optimistic about generalisation beyond this dataset. "
    "The smallest possible two-sided exact Wilcoxon p with 10 folds is 0.002. Holm adjustment is over the 10 pairs "
    "listed in this table only. Differences of a few points between models with overlapping per-fold spread should not "
    "be over-read.",
]


ABLATION_PAIRS = [("a1_mobilenetv3_small_scratch_lr3e4", "mobilenetv3_small_scratch"),
                  ("a1_mobilenetv3_small_scratch_lr3e4", "mobilenetv3_small_imagenet")]


def ablation_table(results_dir="results", metric="accuracy") -> pd.DataFrame:
    """A-1 against the two main MobileNetV3-S runs (same folds, paired). Only the two declared pairs."""
    from .aggregate import load_ablations
    data = {**load_all(results_dir), **load_ablations(results_dir)}
    frames = []
    for ea, eb in ABLATION_PAIRS:
        if ea in data and eb in data:
            frames.append(paired_table(results_dir, [ea, eb], metric, data=data))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


ABLATION_NOTES = [
    "Ablation A-1 was declared before its results existed (D-38): MobileNetV3-Small from scratch at lr 3e-4, everything else "
    "as protocol v2. Compare with the scratch run (lr 1e-3) to see the effect of the learning rate alone, and with the "
    "ImageNet run (lr 3e-4) to see the effect of initialisation alone. Two pre-declared pairs, no multiplicity correction "
    "(Holm column is within each pair, i.e. equals the raw p).",
] + PAIRED_NOTES[:1]
