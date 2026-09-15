"""Aggregate real results into the report table (Report Section 4.5).

    python -m src.aggregate --models cnn,crnn
Reads results/<model>/fold*.json and bench.json. Fails loudly if anything is missing -
there is deliberately no fallback value, so a table can only contain measured numbers.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .models import DISPLAY, MODELS


def load(model, root):
    d = Path(root) / model
    folds = sorted(d.glob("fold*.json"))
    if not folds:
        raise SystemExit(f"no fold results for '{model}' in {d}")
    if not (d / "bench.json").exists():
        raise SystemExit(f"missing {d/'bench.json'} - run: python -m src.bench --model {model}")
    return [json.loads(f.read_text()) for f in folds], json.loads((d / "bench.json").read_text())


def pm(vals, scale=1.0, nd=2):
    v = np.array(vals) * scale
    return f"{v.mean():.{nd}f} ± {v.std():.{nd}f}" if len(v) > 1 else f"{v.mean():.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="cnn,crnn")
    ap.add_argument("--root", default="results")
    args = ap.parse_args()

    rows = ["| Model | Pretrained | Accuracy (%) | Macro-F1 | Params (M) | Size (MB) | CPU latency (ms, 1 thread) |",
            "|---|---|---|---|---|---|---|"]
    notes = []
    for m in args.models.split(","):
        folds, bench = load(m, args.root)
        rows.append(f"| {DISPLAY[m]} | {MODELS[m][2]} | {pm([f['test_accuracy'] for f in folds], 100)} | "
                    f"{pm([f['test_macro_f1'] for f in folds], 1, 3)} | {bench['params_millions']:.2f} | "
                    f"{bench['size_mb_fp32']:.1f} | {bench['latency_1_thread']['median_ms']:.1f} |")
        notes.append(f"- {m}: folds {[f['fold'] for f in folds]}, epoch cap {folds[0]['epoch_cap']}, "
                     f"patience {folds[0]['patience']}, device {folds[0]['device']}, "
                     f"latency machine: {bench['machine']['platform']}")
        # summed confusion matrix per model
        cm = np.sum([f["confusion_matrix"] for f in folds], axis=0)
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.imshow(cm, cmap="Blues"); ax.set_title(f"{DISPLAY[m]} ({MODELS[m][2]}) - summed over folds")
            ax.set_xticks(range(10)); ax.set_yticks(range(10))
            ax.set_xticklabels(folds[0]["classes"], rotation=60, ha="right", fontsize=7)
            ax.set_yticklabels(folds[0]["classes"], fontsize=7)
            ax.set_xlabel("predicted"); ax.set_ylabel("true")
            fig.tight_layout(); fig.savefig(Path(args.root) / m / "confusion.png", dpi=150); plt.close(fig)
        except ImportError:
            notes.append("- matplotlib missing: confusion matrices not drawn")
    text = "\n".join(rows) + "\n\nRun settings:\n" + "\n".join(notes) + "\n"
    (Path(args.root) / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
