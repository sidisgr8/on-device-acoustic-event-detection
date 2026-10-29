"""Post-training 8-bit quantization of trained models, evaluated WITHOUT retraining.

For every finished fold with a best.pt (protocol v2) this builds the variants of
`src/compress/variants.py`, evaluates each on CPU and records accuracy, macro-F1 and file size
separately (D-09). No timing is done here (see scripts/benchmark_latency.py).

  --split val   (default) validation fold only: dry runs and correctness checks (D-20)
  --split test  the final evaluation; each (variant, fold) file is created with mode 'x' so the
                test fold is evaluated once.

    python scripts/quant_eval.py --split test                       # all models, all finished folds
    python scripts/quant_eval.py --experiments cnn_baseline --folds 1 --split val --out /tmp/x --artifacts /tmp/y
"""
import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

from src.compress.common import (CKPT_DIR, RESULTS_DIR, available_folds, best_ckpt, evaluate_runner,
                                 find_config, load_trained_model, model_kind, round_data, sha256)
from src.compress.variants import build_variants, variants_for
from src.data.dataset import FeatureStore
from src.train.loop import GIT_AT_START

EXPERIMENTS = ["cnn_baseline", "mobilenetv3_small_scratch", "mobilenetv3_small_imagenet", "crnn", "ast"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", default=EXPERIMENTS)
    ap.add_argument("--folds", nargs="+", type=int, default=None, help="default: every finished fold")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--out", default=str(RESULTS_DIR / "quant"))
    ap.add_argument("--artifacts", default=str(CKPT_DIR / "compressed"))
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N clips (dry runs)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    import onnxruntime
    store = FeatureStore()
    out = Path(a.out)

    for exp in a.experiments:
        cfg = find_config(exp)
        kind = model_kind(cfg)
        folds = a.folds or available_folds(exp, a.ckpt_dir, a.results_dir)
        if not folds:
            print(f"[{exp}] no finished folds with best.pt, skipping")
            continue
        for k in folds:
            names = variants_for(kind)
            paths = {v: out / exp / v / f"fold{k}.json" for v in names}
            if all(p.exists() for p in paths.values()):
                continue
            try:
                t0 = time.time()
                print(f"[{exp} fold {k}] split={a.split} building variants", flush=True)
                ckpt = best_ckpt(exp, k, a.ckpt_dir)
                model = load_trained_model(cfg, ckpt)
                ds, calib, calib_rows, stats = round_data(store, cfg, k, a.split)
                check_x = ds.x[:16]
                built = build_variants(kind, model, calib, check_x,
                                       Path(a.artifacts) / exp / f"fold{k}", log=print)
                fp32_size = built["fp32_torch"]["runner"].size_bytes
                best_meta = json.loads((ckpt.parent / "best.json").read_text())
                for v in names:
                    if paths[v].exists():  # already written: a test fold is evaluated once
                        continue
                    b = built[v]
                    rec = {"experiment": exp, "variant": v, "fold": k, "split": a.split,
                           "protocol_version": 2, "fp32_size_bytes": fp32_size,
                           "source_best_pt_sha256": sha256(ckpt),
                           "source_best_epoch": best_meta["epoch"],
                           "source_best_val_acc_recorded_on_mps": best_meta["val_acc"],
                           "n_calibration_clips": len(calib), "calibration_rows": calib_rows.tolist(),
                           "torch": torch.__version__, "onnxruntime": onnxruntime.__version__,
                           "threads": a.threads, "git": GIT_AT_START, "date": datetime.now().isoformat(timespec="seconds")}
                    if "error" in b:
                        rec["error"] = b["error"]
                    else:
                        r = b["runner"]
                        m = evaluate_runner(r, ds, limit=a.limit)
                        rec.update(backend=r.backend, accuracy=m["accuracy"], macro_f1=m["macro_f1"],
                                   n_eval=m["n_eval"], size_bytes=r.size_bytes,
                                   size_ratio_vs_fp32=r.size_bytes / fp32_size, meta=r.meta)
                        print(f"    {v:22s} acc {m['accuracy']:.4f} f1 {m['macro_f1']:.4f} "
                              f"size {r.size_bytes / 1e6:.2f} MB", flush=True)
                    paths[v].parent.mkdir(parents=True, exist_ok=True)
                    with open(paths[v], "x" if a.split == "test" else "w") as f:
                        json.dump(rec, f, indent=2)
                print(f"[{exp} fold {k}] done in {time.time() - t0:.0f}s", flush=True)
            except Exception as e:  # record and move on (resumable; finished variants are never re-evaluated)
                traceback.print_exc()
                err = out / exp / f"fold{k}.error.txt"
                err.parent.mkdir(parents=True, exist_ok=True)
                err.write_text(f"{type(e).__name__}: {e}\n")
                print(f"[{exp} fold {k}] FAILED ({type(e).__name__}: {e}); recorded in {err}, continuing", flush=True)


if __name__ == "__main__":
    main()
