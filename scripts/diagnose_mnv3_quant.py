"""Diagnose the int8 collapse of the ImageNet-initialised MobileNetV3-S (VALIDATION data only, D-20).

    .venv-post/bin/python scripts/diagnose_mnv3_quant.py [--folds 1 5 9] [--out results/diagnostics/mnv3_quant_val.json]

For the ImageNet model and the scratch model (comparison) on each fold's checkpoint:
  1. per-layer statistics of the BN-folded conv weights (per-tensor abs max, per-output-channel abs-max spread) and of the
     activation ranges seen on the calibration clips,
  2. validation accuracy of the existing static-int8 setups and of candidate variants (larger calibration set, per-channel
     weights, other activation observers, float first/last layers, float head, ...), for PyTorch FX and ONNX Runtime,
  3. a leave-one-group-in-float sweep that localises where the damage is.
No test fold is touched. Calibration clips come from the training folds (same seed as the main run).
"""
import argparse
import copy
import json
import warnings
from pathlib import Path

import numpy as np
import torch

from src.compress.common import best_ckpt, find_config, load_trained_model, round_data
from src.compress.variants import ENGINE, export_onnx, ort_session
from src.data.dataset import FeatureStore
from src.eval.metrics import classification_metrics

EXPS = ["mobilenetv3_small_imagenet", "mobilenetv3_small_scratch"]


def val_metrics(fn, ds, bs=32):
    preds = []
    for s in range(0, len(ds), bs):
        with torch.no_grad():
            preds.append(fn(ds.x[s:s + bs]).argmax(1))
    m = classification_metrics(ds.y.numpy(), torch.cat(preds).numpy())
    return {"val_acc": m["accuracy"], "val_macro_f1": m["macro_f1"]}


# ------------------------------------------------------------------ PyTorch FX candidates
def fx_qconfig(per_channel=False, act="hist"):
    from torch.ao.quantization import (HistogramObserver, MinMaxObserver, MovingAverageMinMaxObserver,
                                       PerChannelMinMaxObserver, QConfig)
    a = {"hist": HistogramObserver.with_args(reduce_range=False),
         "minmax": MinMaxObserver.with_args(reduce_range=False),
         "ma_minmax": MovingAverageMinMaxObserver.with_args(reduce_range=False)}[act]
    w = (PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric) if per_channel
         else MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
    return QConfig(activation=a, weight=w)


def fx_build(model, calib_x, per_channel=False, act="hist", float_regex=(), return_prepared=False):
    from torch.ao.quantization import QConfigMapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    torch.backends.quantized.engine = ENGINE
    qmap = QConfigMapping().set_global(fx_qconfig(per_channel, act))
    for rg in float_regex:
        qmap.set_module_name_regex(rg, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = prepare_fx(copy.deepcopy(model).eval(), qmap, (calib_x[:1],))
        with torch.no_grad():
            for s in range(0, len(calib_x), 10):
                prepared(calib_x[s:s + 10])
        if return_prepared:
            return prepared
        q = convert_fx(prepared)
    return lambda x: q(x)


# ------------------------------------------------------------------ ONNX candidates
def onnx_build(model, calib_x, workdir, tag, per_channel=False, method="minmax", exclude_first_last=False):
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    import onnx
    workdir.mkdir(parents=True, exist_ok=True)
    fp32 = workdir / "model_fp32.onnx"
    if not fp32.exists():
        export_onnx(model, calib_x[:1], fp32)
    pre = workdir / "model_pre.onnx"
    if not pre.exists():
        try:
            quant_pre_process(str(fp32), str(pre))
        except Exception:
            pre = fp32

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"input": calib_x[i:i + 1].numpy()} for i in range(len(calib_x))])

        def get_next(self):
            return next(self.it, None)

    excl = []
    if exclude_first_last:
        nodes = onnx.load(str(pre)).graph.node
        convs = [n.name for n in nodes if n.op_type == "Conv"]
        last = [n.name for n in nodes if n.op_type in ("Gemm", "MatMul")]
        excl = [convs[0]] + last[-1:]
    cm = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
          "percentile": CalibrationMethod.Percentile}[method]
    out = workdir / f"int8_{tag}.onnx"
    extra = {"CalibPercentile": 99.99} if method == "percentile" else {}
    quantize_static(str(pre), str(out), Reader(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=per_channel, calibrate_method=cm,
                    nodes_to_exclude=excl, extra_options=extra)
    sess = ort_session(out, threads=2)
    return lambda x: torch.from_numpy(sess.run(None, {"input": x.numpy()})[0]), excl


# ------------------------------------------------------------------ statistics
def layer_stats(model, calib_x):
    """BN-folded weight and activation-range statistics per conv/linear layer (from an FX-prepared copy, eval mode)."""
    prepared = fx_build(model, calib_x, return_prepared=True)
    rows = []
    obs = {n: m for n, m in prepared.named_modules() if "activation_post_process" in n and hasattr(m, "min_val")}
    for name, m in prepared.named_modules():
        w = getattr(m, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() >= 2 and "activation_post_process" not in name:
            wd = w.detach().float()
            ch = wd.abs().flatten(1).amax(1)                       # per-output-channel abs max (BN already folded)
            rows.append({"layer": name, "kind": type(m).__name__, "shape": list(w.shape), "w_absmax": float(wd.abs().max()),
                         "w_channel_absmax_max_over_median": float(ch.max() / ch.median().clamp_min(1e-12)),
                         "w_channel_absmax_max_over_min": float(ch.max() / ch.min().clamp_min(1e-12)),
                         "w_per_tensor_int8_step_over_median_channel_absmax": float(wd.abs().max() / 127 / ch.median().clamp_min(1e-12))})
    acts = [{"observer": n, "min": float(m.min_val.min()), "max": float(m.max_val.max())} for n, m in obs.items()]
    return rows, acts


def summarize_stats(rows, acts):
    ratios = np.array([r["w_channel_absmax_max_over_median"] for r in rows])
    amax = np.array([max(abs(a["min"]), abs(a["max"])) for a in acts]) if acts else np.array([0.0])
    return {"n_layers": len(rows),
            "weight_channel_absmax_ratio_max_over_median": {"median": float(np.median(ratios)), "p90": float(np.percentile(ratios, 90)),
                                                           "max": float(ratios.max()),
                                                           "worst_layer": rows[int(ratios.argmax())]["layer"]},
            "weight_absmax_max": float(max(r["w_absmax"] for r in rows)),
            "activation_absmax": {"median": float(np.median(amax)), "p90": float(np.percentile(amax, 90)), "max": float(amax.max())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 5, 9])
    ap.add_argument("--out", default="results/diagnostics/mnv3_quant_val.json")
    ap.add_argument("--work", default="checkpoints/diagnostics")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    store = FeatureStore()
    result = {"split": "validation only (D-20)", "engine": ENGINE, "folds": a.folds, "models": {}}
    float_groups = {"stem": [r"net\.conv_stem", r"net\.bn1"], "head": [r"net\.conv_head", r"net\.classifier"],
                    "classifier": [r"net\.classifier"]}
    for i in range(6):
        float_groups[f"blocks.{i}"] = [rf"net\.blocks\.{i}\..*", rf"net\.blocks\.{i}"]
    for exp in EXPS:
        cfg = find_config(exp)
        res = result["models"].setdefault(exp, {})
        for k in a.folds:
            model = load_trained_model(cfg, best_ckpt(exp, k))
            ds, calib100, _, _ = round_data(store, cfg, k, "val", n_calib=100)
            _, calib500, _, _ = round_data(store, cfg, k, "val", n_calib=500)
            r = res.setdefault(f"fold{k}", {"val_split_fold": (k % 10) + 1, "n_val": len(ds)})
            r["fp32"] = val_metrics(lambda x: model(x), ds)
            rows, acts = layer_stats(model, calib100)
            r["stats"] = {"summary": summarize_stats(rows, acts), "layers": rows, "activations": acts}
            cands = {
                "fx_existing(100 clips, per-tensor, histogram)": dict(),
                "fx_calib500": dict(calib=500),
                "fx_per_channel": dict(per_channel=True),
                "fx_act_minmax": dict(act="minmax"),
                "fx_act_moving_avg_minmax": dict(act="ma_minmax"),
                "fx_float_first_last": dict(float_regex=float_groups["stem"] + float_groups["classifier"]),
                "fx_float_head": dict(float_regex=float_groups["head"]),
                "fx_per_channel+calib500": dict(per_channel=True, calib=500),
                "fx_per_channel+calib500+float_first_last": dict(per_channel=True, calib=500,
                                                                 float_regex=float_groups["stem"] + float_groups["classifier"]),
            }
            r["candidates"] = {}
            for name, kw in cands.items():
                kw = dict(kw)
                cx = calib500 if kw.pop("calib", 100) == 500 else calib100
                try:
                    r["candidates"][name] = val_metrics(fx_build(model, cx, **kw), ds)
                except Exception as e:
                    r["candidates"][name] = {"error": f"{type(e).__name__}: {e}"[:300]}
                print(exp, k, name, r["candidates"][name], flush=True)
            r["float_one_group_sweep"] = {}                        # existing setup, but ONE group kept in float
            for g, rgx in float_groups.items():
                try:
                    r["float_one_group_sweep"][g] = val_metrics(fx_build(model, calib100, float_regex=rgx), ds)
                except Exception as e:
                    r["float_one_group_sweep"][g] = {"error": f"{type(e).__name__}: {e}"[:300]}
                print(exp, k, "float", g, r["float_one_group_sweep"][g], flush=True)
            wd = Path(a.work) / exp / f"fold{k}"
            r["onnx"] = {}
            ocands = {"onnx_existing(100 clips, per-tensor, minmax)": dict(),
                      "onnx_per_channel": dict(per_channel=True), "onnx_calib500": dict(calib=500),
                      "onnx_percentile": dict(method="percentile"), "onnx_entropy": dict(method="entropy"),
                      "onnx_exclude_first_last": dict(exclude_first_last=True),
                      "onnx_per_channel+calib500": dict(per_channel=True, calib=500),
                      "onnx_per_channel+calib500+exclude_first_last": dict(per_channel=True, calib=500, exclude_first_last=True)}
            for j, (name, kw) in enumerate(ocands.items()):
                kw = dict(kw)
                cx = calib500 if kw.pop("calib", 100) == 500 else calib100
                try:
                    fn, excl = onnx_build(model, cx, wd, f"c{j}", **kw)
                    r["onnx"][name] = {**val_metrics(fn, ds), "excluded_nodes": excl}
                except Exception as e:
                    r["onnx"][name] = {"error": f"{type(e).__name__}: {e}"[:300]}
                print(exp, k, name, r["onnx"][name], flush=True)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(result, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
