"""One-off precision probe for AST (D-11 / D-27): fp32 vs fp16 (+GradScaler) vs bf16.

Runs ~30 optimiser steps of AST fine-tuning on fold-1 TRAINING clips only (no validation or test
data is touched), with identical weights, batches, augmentation masks and dropout seeds in every
mode. Rule, fixed in advance: a reduced-precision mode is eligible if every loss is finite and the
mean of its last 10 losses is within 0.10 of the fp32 value; the fastest eligible mode (median step
time after 3 warm-up steps) is chosen; fp32 is always eligible and is the fallback if anything fails.
"""
import copy
import json
import statistics as st
import time
from datetime import date
from pathlib import Path

import torch

from src.data.dataset import FeatureStore, SpecAugment, BatchDataset, train_loader
from src.data.splits import round_indices
from src.features import params as P
from src.models import build_model

TOL = 0.10
WARMUP = 3


def select_mode(modes: dict) -> str:
    """Fastest eligible mode; fp32 if nothing else qualifies (fp32 is the reference)."""
    eligible = [m for m, r in modes.items() if r["eligible"]]
    return min(eligible, key=lambda m: modes[m]["median_step_s"]) if eligible else "fp32"


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def _run_mode(base, mode, ds, cfg, device, steps):
    tcfg = cfg["train"]
    torch.manual_seed(tcfg["seed"])
    model = copy.deepcopy(base).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    crit = torch.nn.CrossEntropyLoss(label_smoothing=tcfg["label_smoothing"])
    aug = SpecAugment(**cfg["specaugment"], freq_dim=2, time_dim=1,
                      generator=torch.Generator().manual_seed(tcfg["seed"])).train()
    dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[mode]
    scaler = torch.amp.GradScaler(device.type) if mode == "fp16" else None
    torch.manual_seed(tcfg["seed"] * 1000 + 1)
    losses, times = [], []
    for i, (x, y, _) in enumerate(train_loader(ds, tcfg["batch_size"], tcfg["seed"], 1)):
        if i >= steps:
            break
        x, y = aug(x).to(device), y.to(device)
        _sync(device)
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            logits = model(x)
        loss = crit(logits.float(), y)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        _sync(device)
        times.append(time.perf_counter() - t0)
        losses.append(loss.item())
    mem = torch.mps.driver_allocated_memory() / 1e9 if device.type == "mps" else None
    del model, opt
    if device.type == "mps":
        torch.mps.empty_cache()
    return {"losses": losses, "finite": all(map(lambda v: v == v and abs(v) != float("inf"), losses)),
            "median_step_s": st.median(times[WARMUP:]), "n_steps": len(losses), "mps_gb": mem}


def run_probe(cfg: dict, store: FeatureStore, device, out_path, decisions_path=None, steps=30,
              log=print) -> dict:
    out_path = Path(out_path)
    result = {"rule": f"finite losses and |mean(last10) - fp32 mean(last10)| <= {TOL}; fastest eligible; "
                      "fp32 fallback", "steps": steps, "data": "fold-1 TRAINING clips only", "modes": {}}
    try:
        tr, _, _ = round_indices(store.fold, 1)
        ds = BatchDataset(store.ast()[tr], store.labels[tr], tr)
        base = build_model({**cfg["model"]})
        ref = None
        for mode in ("fp32", "fp16", "bf16"):
            r = _run_mode(base, mode, ds, cfg, device, steps)
            r["last10_mean"] = st.mean(r["losses"][-10:])
            if mode == "fp32":
                ref = r["last10_mean"]
                r["eligible"] = r["finite"]
            else:
                r["delta_vs_fp32"] = r["last10_mean"] - ref
                r["eligible"] = r["finite"] and abs(r["delta_vs_fp32"]) <= TOL
            result["modes"][mode] = r
            log(f"  amp probe {mode}: finite={r['finite']} last10={r['last10_mean']:.4f} "
                f"step={r['median_step_s'] * 1000:.0f} ms eligible={r['eligible']}")
        result["chosen"] = select_mode(result["modes"])
        result["reason"] = "fastest eligible mode"
    except Exception as e:  # anything failing -> fp32, and say so
        result["chosen"] = "fp32"
        result["reason"] = f"probe failed, fell back to fp32: {e!r}"
        log(f"  amp probe FAILED: {e!r}; falling back to fp32")
    out_path.write_text(json.dumps(result, indent=2))
    if decisions_path:
        m = result["modes"]
        ev = "; ".join(f"{k}: finite={v['finite']}, last-10 loss {v['last10_mean']:.3f}, {v['median_step_s'] * 1000:.0f} ms/step"
                       for k, v in m.items()) or "no measurements"
        row = (f"| D-28 | {date.today().isoformat()} | Decided (automatic, by the queue's amp_probe stage) | "
               f"AST precision = **{result['chosen']}**. {result['reason']}. Evidence (30 steps, fold-1 training "
               f"clips only, results/ast_amp_probe.json): {ev}. | Rule fixed in advance (D-27). |\n")
        with open(decisions_path, "a") as f:
            f.write(row)
    return result
