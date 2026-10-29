"""Latency / efficiency benchmark. MANUAL stage: needs an otherwise idle machine.

    python scripts/benchmark_latency.py            # all models and every compressed variant found on disk

Refuses to run (exit code 2) if a training/post-processing job is running or the load average is high.
There is deliberately no override flag. Writes results/latency/latency.json.

The whole benchmark is repeated `--repeats` times (default 3) with a different model order each time
(given, reversed, rotated); reported numbers are the median across repeats of each repeat's median/p95, with the
spread across repeats (flagged above 10 %). Only fold 1's weights/artifacts are timed (latency does not depend on
the weights; D-40), and one pruned variant per model (`--pruned-sparsity`, default 0.7).

Per model and variant: forward pass at 1 and 4 threads (batch 1, one 4 s validation clip, 20 warm-up +
200 timed runs, median and p95); feature extraction (log-mel or AST fbank + normalisation) is timed
separately, also at 1 and 4 threads. Variants come from checkpoints/compressed (quantization artifacts)
and checkpoints/pruned (fp32 eager with pruned, still dense, weights).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.benchmark.latency import (BUSY_PATTERNS, RUNS, THREADS, WARMUP, aggregate_repeats, check_idle, machine_metadata,
                                    machine_state, measure, repeat_orders, summarize)
from src.compress.common import (CKPT_DIR, RESULTS_DIR, best_ckpt, find_config, load_trained_model,
                                 model_kind)
from src.compress.variants import load_runner, variants_for
from src.data.dataset import FeatureStore
from src.features import params as P
from src.features.logmel import apply_floor, ast_fbank, ast_normalize, logmel_db
from src.train.loop import GIT_AT_START

EXPERIMENTS = ["cnn_baseline", "mobilenetv3_small_scratch", "mobilenetv3_small_imagenet", "crnn", "ast"]


def feature_fn(cfg, wav, norm):
    if cfg["data"].get("input", "logmel") == "ast":
        def f():
            return torch.from_numpy(ast_normalize(ast_fbank(wav))).unsqueeze(0)
    else:
        floor, mean, std = cfg["data"]["floor_db"], norm["mean"], norm["std"]

        def f():
            x = apply_floor(logmel_db(wav), floor)
            return torch.from_numpy((x - mean) / std).float()[None, None]
    return f


def run_once(a, order, wav, sp_name):
    records, feat_records = [], []
    for exp in order:
        cfg = find_config(exp)
        kind = model_kind(cfg)
        if not best_ckpt(exp, a.fold, a.ckpt_dir).exists():
            print(f"[{exp}] no best.pt for fold {a.fold}, skipping")
            continue
        norm = json.loads((Path(a.results_dir) / exp / f"fold{a.fold}.json").read_text())["norm"]
        feat = feature_fn(cfg, wav, norm)
        model = load_trained_model(cfg, best_ckpt(exp, a.fold, a.ckpt_dir))
        for th in THREADS:
            torch.set_num_threads(th)
            feat_records.append({"experiment": exp, "threads": th, "stage": "feature_extraction",
                                 **summarize(measure(feat))})
        x = feat()
        variants = [(v, {}) for v in variants_for(kind)]
        pruned_root = Path(a.pruned) / exp
        if pruned_root.exists():
            variants += [(f"pruned_{d.name}", {"pruned": d}) for d in sorted(pruned_root.glob(sp_name))
                         if (d / f"fold{a.fold}" / "pruned.pt").exists()]
        for v, extra in variants:
            for th in THREADS:
                torch.set_num_threads(th)
                try:
                    if "pruned" in extra:
                        m = load_trained_model(cfg, best_ckpt(exp, a.fold, a.ckpt_dir))
                        m.load_state_dict(torch.load(extra["pruned"] / f"fold{a.fold}" / "pruned.pt"))
                        from src.compress.variants import fp32_runner
                        runner = fp32_runner(m)
                    else:
                        runner = load_runner(v, kind, Path(a.artifacts) / exp / f"fold{a.fold}",
                                             model_fp32=model, threads=th)
                    with torch.no_grad():
                        t = measure(lambda: runner(x))
                    rec = {"experiment": exp, "variant": v, "fold": a.fold, "threads": th, "stage": "forward",
                           "size_bytes": runner.size_bytes, **summarize(t)}
                    print(f"{exp:28s} {v:22s} {th}T  median {rec['median_ms']:8.2f} ms  p95 {rec['p95_ms']:8.2f} ms", flush=True)
                except Exception as e:
                    rec = {"experiment": exp, "variant": v, "fold": a.fold, "threads": th, "stage": "forward",
                           "error": f"{type(e).__name__}: {e}"[:400]}
                    print(f"{exp:28s} {v:22s} {th}T  FAILED {rec['error'][:120]}", flush=True)
                records.append(rec)
    return records, feat_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", default=EXPERIMENTS)
    ap.add_argument("--fold", type=int, default=1, help="whose weights/artifacts to time (latency does not depend on the weights)")
    ap.add_argument("--repeats", type=int, default=3, help="full repeats with a different model order each time")
    ap.add_argument("--pruned-sparsity", type=float, default=0.7, help="the one pruned variant timed per model")
    ap.add_argument("--out", default=str(RESULTS_DIR / "latency" / "latency.json"))
    ap.add_argument("--artifacts", default=str(CKPT_DIR / "compressed"))
    ap.add_argument("--pruned", default=str(CKPT_DIR / "pruned"))
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    a = ap.parse_args()

    reasons = check_idle()
    if reasons:
        print("REFUSING to benchmark: the machine is not idle:\n  - " + "\n  - ".join(reasons), file=sys.stderr)
        sys.exit(2)

    store = FeatureStore()
    from src.data.splits import round_indices
    _, va, _ = round_indices(store.fold, a.fold)
    wav_all = np.load(store.cache / "waveforms.npy", mmap_mode="r")
    wav = np.asarray(wav_all[va[0]])                        # one 4 s validation clip
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    orders = repeat_orders(a.experiments, a.repeats)
    sp_name = f"s{int(round(a.pruned_sparsity * 100))}"
    states, all_fwd, all_feat = [machine_state()], [], []
    for r, order in enumerate(orders):
        if r > 0:                                            # processes only: our own earlier repeat leaves the load average high
            busy = [x for x in check_idle(max_load=1e9) if "process running" in x]
            if busy:
                print("REFUSING to continue, something else started: " + "; ".join(busy), file=sys.stderr)
                sys.exit(2)
        print(f"=== repeat {r + 1}/{len(orders)}  order: {order}", flush=True)
        states.append({"repeat": r + 1, "start": machine_state()})
        records, feat_records = run_once(a, order, wav, sp_name)
        states[-1]["end"] = machine_state()
        for x in records + feat_records:
            x["repeat"] = r + 1
        all_fwd.append(records)
        all_feat.append(feat_records)
    states.append(machine_state())
    meta = machine_metadata({"git": GIT_AT_START, "clip": "first validation clip of fold %d" % a.fold,
                             "idle_check": "passed (processes and 1-min load < 2.0 at start (D-42))",
                             "weights_used": f"fold {a.fold} only (D-40): latency does not depend on the trained weights",
                             "pruned_variant": f"{sp_name} only (unstructured sparsity, dense fp32 eager execution)",
                             "repeats": len(orders), "repeat_orders": orders,
                             "aggregation": "median over repeats of each repeat's median / p95; spread_pct = (max-min)/median of the repeat medians; noisy = spread > 10 %",
                             "thread_settings": "torch.set_num_threads(n) and ONNX Runtime intra_op_num_threads=n; n in (1, 4); OMP_NUM_THREADS unset",
                             "machine_states": states})
    fwd, feat = aggregate_repeats(all_fwd), aggregate_repeats(all_feat)
    noisy = [(x["experiment"], x.get("variant", "feature_extraction"), x["threads"], round(x["spread_pct"], 1))
             for x in fwd + feat if x.get("noisy")]
    out_path.write_text(json.dumps({"meta": meta, "forward": fwd, "feature_extraction": feat,
                                    "repeats_raw": {"forward": all_fwd, "feature_extraction": all_feat},
                                    "noisy_measurements": noisy}, indent=2))
    print(f"measurements with spread across repeats > 10 %: {noisy or 'none'}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
