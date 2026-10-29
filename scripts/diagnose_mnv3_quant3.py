"""Third round of the MobileNetV3-S int8 diagnosis (VALIDATION only): weights or activations?

Simulated (fake) 8-bit quantization with the same 100 calibration clips, per-tensor, qnnpack-style ranges:
weights only / activations only / both, for the ImageNet and the scratch model. Fake quantization is a simulation of the
int8 arithmetic (no integer kernels), used here only to separate the two error sources.
"""
import json
from pathlib import Path

import numpy as np
import torch

from scripts.diagnose_mnv3_quant import EXPS, val_metrics
from src.compress.common import best_ckpt, find_config, load_trained_model, round_data
from src.data.dataset import FeatureStore


def fake_quant_model(model, calib_x):
    """Calibrated FX-prepared copy (BN folded, histogram activation observers, as in the existing setup) with switchable
    simulated int8: per-tensor symmetric weights, per-tensor affine uint8 activations at the observers' qparams."""
    import copy
    from scripts.diagnose_mnv3_quant import fx_build
    prepared = fx_build(model, calib_x, return_prepared=True)
    w_orig = {}
    for n, m in prepared.named_modules():
        w = getattr(m, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() >= 2 and "activation_post_process" not in n:
            w_orig[n] = w.detach().clone()
    obs = {n: m for n, m in prepared.named_modules() if "activation_post_process" in n and hasattr(m, "calculate_qparams")}
    qp = {n: m.calculate_qparams() for n, m in obs.items()}
    fwd = {n: m.forward for n, m in obs.items()}

    def run(mode):
        mods = dict(prepared.named_modules())
        for n, w in w_orig.items():
            if mode in ("weights", "both"):
                s = w.abs().max() / 127
                mods[n].weight.data = torch.round(w / s).clamp(-128, 127) * s
            else:
                mods[n].weight.data = w.clone()
        for n, m in obs.items():
            if mode in ("activations", "both"):
                scale, zp = qp[n]
                lo, hi = (0, 255)
                m.forward = (lambda x, scale=scale, zp=zp: torch.fake_quantize_per_tensor_affine(
                    x, float(scale), int(zp), lo, hi))
            else:
                m.forward = lambda x: x
        return lambda x: prepared(x)
    return run, len(obs)


def main():
    torch.set_num_threads(2)
    store = FeatureStore()
    out = {"split": "validation only (D-20)", "note": "simulated (fake) int8, per-tensor, 100 calibration clips", "models": {}}
    for exp in EXPS:
        cfg = find_config(exp)
        res = out["models"].setdefault(exp, {})
        for k in (1, 5, 9):
            model = load_trained_model(cfg, best_ckpt(exp, k))
            ds, calib, _, _ = round_data(store, cfg, k, "val", n_calib=100)
            fp = val_metrics(lambda x: model(x), ds)
            run, n = fake_quant_model(model, calib)
            r = res.setdefault(f"fold{k}", {"fp32": fp, "n_fake_quant_modules": n})
            for mode in ("none", "weights", "activations", "both"):
                r[mode] = val_metrics(run(mode), ds)
            print(exp, k, {m: round(r[m]["val_acc"] * 100, 1) for m in ("fp32", "none", "weights", "activations", "both")}, flush=True)
    Path("results/diagnostics/mnv3_quant_val_weights_vs_activations.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
