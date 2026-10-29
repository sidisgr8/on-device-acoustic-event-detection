"""Plain-text report of the MobileNetV3-S int8 / pruning diagnostics and of the pruning controls (read-only on result files).

    .venv-post/bin/python scripts/report_diagnostics.py [--part quant|prune_recovery|controls|all]

quant, prune_recovery: VALIDATION data only (results/diagnostics/*.json).
controls: test-fold results of the pruning runs, 0 %-sparsity control and the matched-fold unpruned baselines (results/prune, results/<exp>).
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

D = Path("results/diagnostics")
EXPS = {"mobilenetv3_small_imagenet": "ImageNet MobileNetV3-S", "mobilenetv3_small_scratch": "Scratch MobileNetV3-S"}
P = lambda x: f"{100 * x:5.1f}"


def quant():
    d = json.loads((D / "mnv3_quant_val.json").read_text())
    print("=" * 100)
    print("INT8 DIAGNOSIS, VALIDATION DATA ONLY (each fold's validation fold = (k mod 10)+1). Accuracy in %.")
    print("Calibration clips: 100 (500 where stated) from the TRAINING folds, same seed. Torch FX engine: qnnpack. ONNX Runtime 1.31.")
    for exp, name in EXPS.items():
        res = d["models"][exp]
        folds = list(res)
        print("\n" + "-" * 100 + f"\n{name}  ({exp})\n" + "-" * 100)
        print("\nA. fp32 validation accuracy:", "  ".join(f"{f}: {P(res[f]['fp32']['val_acc'])} (val fold {res[f]['val_split_fold']}, n={res[f]['n_val']})" for f in folds))
        print("\nB. Weight and activation range statistics (BN-folded conv/linear weights of the FX-prepared fp32 model; 54 layers)")
        print("   ratio = (largest per-output-channel abs-max) / (median per-output-channel abs-max) of one layer; 1 = all channels equal")
        for f in folds:
            s = res[f]["stats"]["summary"]
            r, a = s["weight_channel_absmax_ratio_max_over_median"], s["activation_absmax"]
            print(f"   {f}: weight ratio over layers: median {r['median']:.2f}, p90 {r['p90']:.2f}, max {r['max']:.3g} (layer {r['worst_layer']}); "
                  f"largest |weight| {s['weight_absmax_max']:.2f}; activation abs-max over the 100 observers: median {a['median']:.1f}, p90 {a['p90']:.1f}, max {a['max']:.1f}")
        f0 = folds[0]
        acts = sorted(res[f0]["stats"]["activations"], key=lambda x: -max(abs(x["min"]), abs(x["max"])))[:6]
        print(f"   largest activation ranges, {f0}: " + "; ".join(f"{a['observer'].split('activation_post_process')[-1]}: [{a['min']:.1f}, {a['max']:.1f}]" for a in acts))
        lay = sorted(res[f0]["stats"]["layers"], key=lambda x: -x["w_channel_absmax_max_over_median"])[:4]
        print(f"   largest per-channel weight spread, {f0}: " + "; ".join(f"{l['layer']}: ratio {l['w_channel_absmax_max_over_median']:.3g}, |w|max {l['w_absmax']:.2f}" for l in lay))
        print("\nC. Remedies tried, VALIDATION accuracy per fold  (fp32: " + "  ".join(P(res[f]["fp32"]["val_acc"]) for f in folds) + ")")
        print(f"   {'PyTorch FX static int8':58s}" + "".join(f"{f:>8s}" for f in folds) + "    mean")
        for n in res[folds[0]]["candidates"]:
            v = [res[f]["candidates"][n].get("val_acc", np.nan) for f in folds]
            print(f"   {n:58s}" + "".join(f"{P(x):>8s}" for x in v) + f"  {P(np.nanmean(v))}")
        print(f"   {'ONNX Runtime static int8 (QDQ)':58s}")
        for n in res[folds[0]]["onnx"]:
            v = [res[f]["onnx"][n].get("val_acc", np.nan) for f in folds]
            print(f"   {n:58s}" + "".join(f"{P(x):>8s}" for x in v) + f"  {P(np.nanmean(v))}")
        print("\nD. Leave-one-group-in-float sweep (existing setup, ONLY the named group kept in float)")
        for g in res[folds[0]]["float_one_group_sweep"]:
            v = [res[f]["float_one_group_sweep"][g].get("val_acc", np.nan) for f in folds]
            print(f"   {g:58s}" + "".join(f"{P(x):>8s}" for x in v) + f"  {P(np.nanmean(v))}")
    t = json.loads((D / "mnv3_quant_val_types.json").read_text())
    print("\n" + "-" * 100 + "\nE. Module-type sweep (existing setup, ONLY the named module type kept in float). 'everything float' must equal fp32.\n" + "-" * 100)
    for exp, name in EXPS.items():
        res = t["models"][exp]
        folds = list(res)
        print(f"\n{name}  fp32: " + "  ".join(P(res[f]["fp32"]["val_acc"]) for f in folds))
        for g in [k for k in res[folds[0]] if k != "fp32"]:
            v = [res[f][g].get("val_acc", np.nan) for f in folds]
            print(f"   {g:58s}" + "".join(f"{P(x):>8s}" for x in v) + f"  {P(np.nanmean(v))}")
    w = json.loads((D / "mnv3_quant_val_weights_vs_activations.json").read_text())
    print("\n" + "-" * 100 + "\nF. Simulated int8, weights only / activations only (per-tensor; 'none' must equal fp32)\n" + "-" * 100)
    for exp, name in EXPS.items():
        res = w["models"][exp]
        print(f"\n{name}")
        for m in ("fp32", "none", "weights", "activations", "both"):
            v = [res[f][m]["val_acc"] for f in res]
            print(f"   {m:12s}" + "".join(f"{P(x):>8s}" for x in v) + f"  {P(np.mean(v))}")


def prune_recovery():
    p = D / "prune_recovery_val.json"
    if not p.exists():
        print("no prune recovery results yet")
        return
    d = json.loads(p.read_text())
    print("=" * 100)
    print("PRUNING RECOVERY DIAGNOSTIC, ImageNet MobileNetV3-S, VALIDATION data only (accuracy %). Main run = 0.1 x lr (3e-5), 3 epochs.")
    for r in d["runs"]:
        pass
    keys = sorted({(r["sparsity"], r["lr"], r["epochs"], bool(r.get("main_run_reference"))) for r in d["runs"]})
    ex = "mobilenetv3_small_imagenet"
    unpruned = {}
    for k in (1, 5, 9):
        j = json.loads(Path(f"results/{ex}/fold{k}.json").read_text())
        # validation accuracy of the unpruned model: from the FX-sanity numbers (fp32 val acc) in the diagnostic file
        unpruned[k] = json.loads((D / "mnv3_quant_val.json").read_text())["models"][ex][f"fold{k}"]["fp32"]["val_acc"]
    print("unpruned validation accuracy:", "  ".join(f"fold{k}: {P(v)}" for k, v in unpruned.items()))
    for sp in (0.5, 0.7):
        print(f"\nsparsity {sp:.0%}")
        print(f"   {'fine-tune':32s}" + "".join(f"{'fold' + str(k):>9s}" for k in (1, 5, 9)) + "     mean   (before fine-tune, mean)")
        rows = {}
        for r in d["runs"]:
            if r["sparsity"] == sp:
                rows.setdefault((r["lr"], r["epochs"], bool(r.get("main_run_reference"))), {})[r["fold"]] = r
        for (lr, ep, ref), by in sorted(rows.items(), key=lambda x: (not x[0][2], x[0][1], x[0][0])):
            v = [by[k]["val_acc_best"] for k in (1, 5, 9) if k in by]
            b = [by[k]["val_acc_before_finetune"] for k in (1, 5, 9) if k in by]
            lab = f"lr {lr:.0e}, {ep} epochs" + ("  [MAIN RUN]" if ref else "")
            print(f"   {lab:32s}" + "".join(f"{P(by[k]['val_acc_best']) if k in by else '   -':>9s}" for k in (1, 5, 9)) + f"  {P(np.mean(v))}   ({P(np.mean(b))})")
    print("\nruns completed:", sum(1 for r in d["runs"] if not r.get("main_run_reference")), "of 18 new fine-tunes (3 configs x 2 sparsities x 3 folds)")


def controls():
    print("=" * 100)
    print("PRUNING CONTROLS AND MATCHED-FOLD BASELINES (TEST folds; folds 1, 5, 9). accuracy % / macro-F1")
    names = {"cnn_baseline": "CNN baseline", "mobilenetv3_small_scratch": "MobileNetV3-S scratch", "mobilenetv3_small_imagenet": "MobileNetV3-S ImageNet",
             "crnn": "CRNN", "ast": "AST"}
    for exp, name in names.items():
        print(f"\n{name}")
        hdr = f"   {'row':32s}" + "".join(f"{'fold' + str(k):>14s}" for k in (1, 5, 9)) + "      mean acc / F1"
        print(hdr)

        def line(label, recs):
            by = {r["fold"]: r for r in recs}
            if not by:
                return
            cells = "".join(f"{(P(by[k]['accuracy']) + ' /' + f'{by[k]['macro_f1']:.3f}') if k in by else '-':>14s}" for k in (1, 5, 9))
            n = [by[k] for k in (1, 5, 9) if k in by]
            print(f"   {label:32s}{cells}      {P(np.mean([r['accuracy'] for r in n]))} / {np.mean([r['macro_f1'] for r in n]):.3f}   (n={len(n)})" + (f"   non-zero params {np.mean([r['nonzero_params'] for r in n]) / 1e6:.3f} M" if "nonzero_params" in n[0] else ""))
        base = [json.loads(Path(f"results/{exp}/fold{k}.json").read_text()) for k in (1, 5, 9)]
        line("unpruned fp32 (matched folds)", base)
        for sp, lab, suf in ((0.0, "0% sparsity fine-tune CONTROL", ""), (0.3, "30% pruned", ""), (0.5, "50% pruned", ""), (0.7, "70% pruned", ""),
                             (0.5, "50% EXTENDED fine-tune recipe", "_ext"), (0.7, "70% EXTENDED fine-tune recipe", "_ext")):
            recs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(f"results/prune/{exp}/s{int(round(sp * 100))}{suf}/fold*.json"))]
            recs = [r for r in recs if not r.get("dry_run") and "accuracy" in r]
            line(lab, recs)
    print("\nEXTENDED recipe = lr 3e-4, 10 epochs (extended fine-tune recipe, lr/epochs chosen on validation data of the ImageNet MobileNetV3-S only; D-45, D-47).")
    print("\nDelta vs unpruned on the same folds (percentage points of accuracy; mean over available folds):")
    for exp, name in names.items():
        base = {k: json.loads(Path(f"results/{exp}/fold{k}.json").read_text())["accuracy"] for k in (1, 5, 9)}
        out = []
        for sp, suf in ((0.0, ""), (0.3, ""), (0.5, ""), (0.7, ""), (0.5, "_ext"), (0.7, "_ext")):
            fs = sorted(glob.glob(f"results/prune/{exp}/s{int(round(sp * 100))}{suf}/fold*.json"))
            ds = [100 * (json.loads(Path(f).read_text())["accuracy"] - base[json.loads(Path(f).read_text())["fold"]]) for f in fs]
            out.append(f"{sp:.0%}{'ext' if suf else ''}: {np.mean(ds):+.2f} (n={len(ds)})" if ds else f"{sp:.0%}{'ext' if suf else ''}: -")
        print(f"   {name:26s}" + "   ".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["quant", "prune_recovery", "controls", "all"])
    a = ap.parse_args()
    if a.part in ("quant", "all"):
        quant()
    if a.part in ("prune_recovery", "all"):
        prune_recovery()
    if a.part in ("controls", "all"):
        controls()
