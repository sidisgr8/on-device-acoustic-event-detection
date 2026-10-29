"""Second round of the MobileNetV3-S int8 diagnosis (VALIDATION only): which module TYPES hold the damage.

Pipeline sanity (everything kept float must reproduce fp32), then keep one module type in float in the existing static-int8
setup: squeeze-excite (SE) modules, depthwise convs, pointwise convs, hardswish/hardsigmoid, and combinations.
"""
import json
from pathlib import Path

import numpy as np
import torch

from scripts.diagnose_mnv3_quant import EXPS, fx_build, val_metrics
from src.compress.common import best_ckpt, find_config, load_trained_model, round_data
from src.data.dataset import FeatureStore

GROUPS = {
    "sanity: everything float": [r".*"],
    "SE modules float": [r".*\.se\..*", r".*\.se"],
    "SE + stem + head float": [r".*\.se\..*", r".*\.se", r"net\.conv_stem", r"net\.bn1", r"net\.conv_head", r"net\.classifier"],
    "depthwise convs float (conv_dw)": [r".*\.conv_dw.*", r".*\.bn2.*"],
    "pointwise expand float (conv_pw)": [r".*\.conv_pw.*", r".*\.bn1.*"],
    "pointwise project float (conv_pwl)": [r".*\.conv_pwl.*", r".*\.bn3.*"],
    "blocks.0 + SE float": [r"net\.blocks\.0\..*", r"net\.blocks\.0", r".*\.se\..*", r".*\.se"],
}


def main():
    torch.set_num_threads(2)
    store = FeatureStore()
    out = {"split": "validation only (D-20)", "models": {}}
    for exp in EXPS:
        cfg = find_config(exp)
        res = out["models"].setdefault(exp, {})
        for k in (1, 5, 9):
            model = load_trained_model(cfg, best_ckpt(exp, k))
            ds, calib, _, _ = round_data(store, cfg, k, "val", n_calib=100)
            r = res.setdefault(f"fold{k}", {"fp32": val_metrics(lambda x: model(x), ds)})
            for name, rg in GROUPS.items():
                try:
                    r[name] = val_metrics(fx_build(model, calib, float_regex=rg), ds)
                except Exception as e:
                    r[name] = {"error": f"{type(e).__name__}: {e}"[:300]}
                print(exp, k, name, r[name], flush=True)
    p = Path("results/diagnostics/mnv3_quant_val_types.json")
    p.write_text(json.dumps(out, indent=1))
    for exp, res in out["models"].items():
        print(exp, "fp32", np.mean([v["fp32"]["val_acc"] for v in res.values()]))
        for name in GROUPS:
            print(f"   {name:40s}", np.round([v[name].get("val_acc", np.nan) * 100 for v in res.values()], 1))


if __name__ == "__main__":
    main()
