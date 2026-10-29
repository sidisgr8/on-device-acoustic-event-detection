"""Build the comparison tables from result files. Pure functions returning pandas DataFrames.

Rules (D-26, D-20):
* only protocol-v2 results; a results/<exp>/ directory containing a fold file with another or missing
  protocol_version, or folds with different config hashes, raises (never silently mixes);
* compression tables use only final TEST-split files (dry runs, validation files and files with errors are
  reported separately, never averaged in);
* std is the sample standard deviation over folds (ddof=1); '-' when only one fold exists.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = [  # fixed order, also the colour-slot order in the figures
    ("cnn_baseline", "CNN baseline", "no"),
    ("mobilenetv3_small_scratch", "MobileNetV3-S (scratch)", "no"),
    ("mobilenetv3_small_imagenet", "MobileNetV3-S (ImageNet)", "ImageNet"),
    ("crnn", "CRNN (CNN+BiGRU)", "no"),
    ("ast", "AST", "ImageNet+AudioSet"),
]
ABLATIONS = [  # declared before their results exist (docs/DECISIONS.md); never part of the main rows, figures or main paired table
    ("a1_mobilenetv3_small_scratch_lr3e4", "[Ablation A-1] MobileNetV3-S (scratch, lr 3e-4)", "no"),
]
ARCH_OF = {"a1_mobilenetv3_small_scratch_lr3e4": "mobilenetv3_small_scratch"}  # same network, so same MACs
NAMES = {e: n for e, n, _ in MODELS + ABLATIONS}
PRETRAINED = {e: p for e, _, p in MODELS + ABLATIONS}
N_FOLDS = 10


class ProtocolMixError(RuntimeError):
    pass


def load_experiment(results_dir, exp) -> list:
    """All finished v2 folds of one experiment, sorted by fold. Raises on mixing."""
    d = Path(results_dir) / exp
    folds, bad = [], []
    for p in sorted(d.glob("fold*.json")):
        if p.name.endswith(".error.json"):
            continue
        r = json.loads(p.read_text())
        if r.get("protocol_version") != 2:
            bad.append(f"{p.name}: protocol_version={r.get('protocol_version')!r}")
        else:
            folds.append(r)
    if bad:
        raise ProtocolMixError(f"{exp}: refusing to aggregate, non-v2 result files present: {bad}")
    hashes = {r["config_hash"] for r in folds}
    if len(hashes) > 1:
        raise ProtocolMixError(f"{exp}: folds were produced by different configs: {sorted(hashes)}")
    return sorted(folds, key=lambda r: r["fold"])


def load_all(results_dir) -> dict:
    out = {}
    for exp, _, _ in MODELS:
        if (Path(results_dir) / exp).exists():
            f = load_experiment(results_dir, exp)
            if f:
                out[exp] = f
    return out


def load_ablations(results_dir) -> dict:
    out = {}
    for exp, _, _ in ABLATIONS:
        if (Path(results_dir) / exp).exists():
            f = load_experiment(results_dir, exp)
            if f:
                out[exp] = f
    return out


def ms(values, scale=1.0, nd=2) -> str:
    v = np.asarray(values, dtype=float) * scale
    if len(v) == 0:
        return "-"
    if len(v) == 1:
        return f"{v[0]:.{nd}f} (n=1)"
    return f"{v.mean():.{nd}f} ± {v.std(ddof=1):.{nd}f}"


def fold_label(folds) -> str:
    ks = [r["fold"] if isinstance(r, dict) else r for r in folds]
    s = f"{len(ks)}/{N_FOLDS}"
    return s if len(ks) == N_FOLDS else f"{s} (folds {','.join(map(str, ks))})"


def fp32_size_mb(results_dir, ckpt_dir, exp):
    """Size of the saved fp32 weights: from the quantization records if present, else best.pt."""
    for p in sorted((Path(results_dir) / "quant" / exp / "fp32_torch").glob("fold*.json")):
        r = json.loads(p.read_text())
        if r.get("split") == "test" and "fp32_size_bytes" in r:
            return r["fp32_size_bytes"] / 1e6
    for p in sorted((Path(ckpt_dir) / exp).glob("fold*/best.pt")):
        return p.stat().st_size / 1e6
    return None


def complexity_lookup(results_dir) -> dict:
    """{experiment: record} from results/complexity.json (scripts/compute_complexity.py), or {} if absent."""
    p = Path(results_dir) / "complexity.json"
    return json.loads(p.read_text())["models"] if p.exists() else {}


def master_table(results_dir="results", ckpt_dir="checkpoints") -> pd.DataFrame:
    data = load_all(results_dir)
    data.update(load_ablations(results_dir))                      # ablation rows last, labelled in the Model column
    lat = latency_lookup(results_dir)
    cx = complexity_lookup(results_dir)
    rows = []
    for exp, folds in data.items():
        acc = [r["accuracy"] for r in folds]
        f1 = [r["macro_f1"] for r in folds]
        size = fp32_size_mb(results_dir, ckpt_dir, exp)
        arch = ARCH_OF.get(exp, exp)
        rows.append({
            "Model": NAMES[exp], "Pretrained": PRETRAINED[exp], "Folds": fold_label(folds),
            "Accuracy (%)": ms(acc, 100), "Macro-F1": ms(f1, 1, 3),
            "Params (M)": f"{folds[0]['n_params'] / 1e6:.3f}",
            "fp32 size (MB)": "-" if size is None else f"{size:.2f}",
            "MACs (M)": f"{cx[arch]['macs_total'] / 1e6:,.1f}" if arch in cx else "-",
            "CPU latency 1T median (ms)": lat.get((exp, "fp32_torch", 1), "-"),
            "CPU latency 4T median (ms)": lat.get((exp, "fp32_torch", 4), "-"),
            "_acc_mean": float(np.mean(acc)), "_f1_mean": float(np.mean(f1)),
            "_macs": cx[arch]["macs_total"] if arch in cx else None, "_params": folds[0]["n_params"],
            "_n_folds": len(folds), "_experiment": exp, "_ablation": exp in dict((e, 1) for e, _, _ in ABLATIONS),
        })
    return pd.DataFrame(rows)


def latency_lookup(results_dir) -> dict:
    p = Path(results_dir) / "latency" / "latency.json"
    out = {}
    if p.exists():
        for r in json.loads(p.read_text())["forward"]:
            if "median_ms" in r:
                out[(r["experiment"], r["variant"], r["threads"])] = f"{r['median_ms']:.2f}"
    return out


def per_class_f1_table(results_dir="results") -> pd.DataFrame:
    data = load_all(results_dir)
    rows = []
    for exp, folds in data.items():
        classes = folds[0]["classes"]
        row = {"Model": NAMES[exp], "Folds": fold_label(folds)}
        for c in classes:
            row[c] = ms([r["per_class_f1"][c] for r in folds], 1, 3)
        rows.append(row)
    return pd.DataFrame(rows)


def padded_vs_full_table(results_dir="results") -> pd.DataFrame:
    """Pooled over folds (clip-weighted); the paired difference full - padded shows reliance on padding."""
    data = load_all(results_dir)
    rows = []
    for exp, folds in data.items():
        agg = {"padded": [0, 0.0], "full": [0, 0.0]}
        for r in folds:
            for k in agg:
                b = r["by_padding"][k]
                if b["n"]:
                    agg[k][0] += b["n"]
                    agg[k][1] += b["accuracy"] * b["n"]
        pa = agg["padded"][1] / max(agg["padded"][0], 1)
        fa = agg["full"][1] / max(agg["full"][0], 1)
        rows.append({"Model": NAMES[exp], "Folds": fold_label(folds),
                     "Padded clips (n)": agg["padded"][0], "Padded acc (%)": f"{100 * pa:.2f}",
                     "Full-length clips (n)": agg["full"][0], "Full acc (%)": f"{100 * fa:.2f}",
                     "Full − padded (pts)": f"{100 * (fa - pa):+.2f}"})
    return pd.DataFrame(rows)


def _read_compress(results_dir, sub, final_only=True):
    root = Path(results_dir) / sub
    recs, skipped = [], []
    for p in sorted(root.rglob("fold*.json")):
        r = json.loads(p.read_text())
        if r.get("protocol_version") != 2:
            raise ProtocolMixError(f"{p}: protocol_version={r.get('protocol_version')!r}")
        if r.get("dry_run") or (final_only and r.get("split", r.get("split_evaluated")) != "test"):
            skipped.append(str(p))
            continue
        recs.append(r)
    return recs, skipped


def quant_table(results_dir="results") -> pd.DataFrame:
    recs, _ = _read_compress(results_dir, "quant")
    base = {(r["experiment"], r["fold"]): r for r in recs if r["variant"] == "fp32_torch" and "accuracy" in r}
    rows = []
    for exp, _, _ in MODELS:
        for variant in ["fp32_torch", "torch_dynamic_int8", "torch_static_int8_fx", "onnx_fp32", "onnx_int8"]:
            rs = [r for r in recs if r["experiment"] == exp and r["variant"] == variant]
            if not rs:
                continue
            ok = [r for r in rs if "accuracy" in r]
            if not ok:
                rows.append({"Model": NAMES[exp], "Variant": variant, "Folds": fold_label(rs),
                             "Accuracy (%)": "failed", "Macro-F1": "-", "Δ acc vs fp32 (pts)": "-",
                             "Size (MB)": "-", "Size vs fp32": "-", "Note": rs[0]["error"][:90]})
                continue
            d = [100 * (r["accuracy"] - base[(exp, r["fold"])]["accuracy"]) for r in ok if (exp, r["fold"]) in base]
            rows.append({"Model": NAMES[exp], "Variant": variant, "Folds": fold_label(ok),
                         "Accuracy (%)": ms([r["accuracy"] for r in ok], 100),
                         "Macro-F1": ms([r["macro_f1"] for r in ok], 1, 3),
                         "Δ acc vs fp32 (pts)": ms(d, 1, 2) if d else "-",
                         "Size (MB)": f"{np.mean([r['size_bytes'] for r in ok]) / 1e6:.2f}",
                         "Size vs fp32": f"{np.mean([r['size_ratio_vs_fp32'] for r in ok]):.2f}×",
                         "Note": ok[0].get("meta", {}).get("quantization", "")[:90]})
    return pd.DataFrame(rows)


EXT_LABEL = "[extended fine-tune recipe (lr and epochs chosen on validation data of the ImageNet MobileNetV3-S only)]"


def prune_table(results_dir="results") -> pd.DataFrame:
    """Pruned rows plus the 0 %-sparsity fine-tune CONTROL (labelled), each beside the unpruned fp32 results on the same folds."""
    recs, _ = _read_compress(results_dir, "prune")
    fp32 = {}
    for exp in {r["experiment"] for r in recs}:
        for r in load_experiment(results_dir, exp):
            fp32[(exp, r["fold"])] = r
    rows = []
    for exp, _, _ in MODELS:
        for sp, rec in sorted({(r["sparsity"], r.get("recipe", "main")) for r in recs if r["experiment"] == exp},
                             key=lambda x: (x[0], x[1] != "main")):
            rs = [r for r in recs if r["experiment"] == exp and r["sparsity"] == sp and r.get("recipe", "main") == rec]
            base = [fp32[(exp, r["fold"])] for r in rs if (exp, r["fold"]) in fp32]
            d = [100 * (r["accuracy"] - fp32[(exp, r["fold"])]["accuracy"]) for r in rs if (exp, r["fold"]) in fp32]
            rows.append({"Model": NAMES[exp], "Sparsity": ("0% (fine-tune control)" if sp == 0 else f"{sp:.0%}") + (f" {EXT_LABEL}" if rec == "extended" else ""),
                         "Folds": fold_label(rs),
                         "Accuracy (%)": ms([r["accuracy"] for r in rs], 100),
                         "Macro-F1": ms([r["macro_f1"] for r in rs], 1, 3),
                         "Unpruned acc, same folds (%)": ms([b["accuracy"] for b in base], 100),
                         "Unpruned macro-F1, same folds": ms([b["macro_f1"] for b in base], 1, 3),
                         "Δ acc vs unpruned (pts)": ms(d, 1, 2) if d else "-",
                         "Non-zero params (M)": f"{np.mean([r['nonzero_params'] for r in rs]) / 1e6:.3f}",
                         "gzip size (MB)": f"{np.mean([r['size_gzip_bytes'] for r in rs]) / 1e6:.2f}",
                         "gzip, unpruned (MB)": f"{np.mean([r['size_gzip_bytes_unpruned_baseline'] for r in rs]) / 1e6:.2f}"})
    return pd.DataFrame(rows)


def latency_table(results_dir="results") -> pd.DataFrame:
    p = Path(results_dir) / "latency" / "latency.json"
    if not p.exists():
        return pd.DataFrame()
    d = json.loads(p.read_text())

    def spread(r):
        return "-" if "spread_pct" not in r else f"{r['spread_pct']:.1f}" + (" (noisy)" if r.get("noisy") else "")

    rows = []
    for r in d["forward"]:
        if "error" in r:
            rows.append({"Model": NAMES.get(r["experiment"], r["experiment"]), "Variant": r["variant"],
                         "Threads": r["threads"], "Median (ms)": "failed", "p95 (ms)": "-", "Size (MB)": "-",
                         "Spread across repeats (%)": "-"})
        else:
            rows.append({"Model": NAMES.get(r["experiment"], r["experiment"]), "Variant": r["variant"],
                         "Threads": r["threads"], "Median (ms)": f"{r['median_ms']:.2f}",
                         "p95 (ms)": f"{r['p95_ms']:.2f}", "Size (MB)": f"{r['size_bytes'] / 1e6:.2f}",
                         "Spread across repeats (%)": spread(r)})
    for r in d.get("feature_extraction", []):
        rows.append({"Model": NAMES.get(r["experiment"], r["experiment"]), "Variant": "feature extraction",
                     "Threads": r["threads"], "Median (ms)": f"{r['median_ms']:.2f}",
                     "p95 (ms)": f"{r['p95_ms']:.2f}", "Size (MB)": "-", "Spread across repeats (%)": spread(r)})
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, title: str, notes=()) -> str:
    cols = [c for c in df.columns if not c.startswith("_")]
    lines = [f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    for n in notes:
        lines += ["", f"*{n}*"]
    return "\n".join(lines) + "\n"
