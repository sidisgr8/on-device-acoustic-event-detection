"""Generic, resumable training loop for the common protocol (spec section 5).

One call to `run_round` trains one cross-validation round: it logs per-epoch validation
metrics, keeps the best-validation-accuracy checkpoint, stops early, and only then evaluates
the test fold, exactly once, writing `<results_dir>/<experiment>/fold<k>.json`.
"""
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data.dataset import (CLASSES, FeatureStore, SpecAugment, eval_loader, make_round,
                              make_round_ast, train_loader)
from src.eval.metrics import classification_metrics, subset_accuracy
from src.models import build_model


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


PROTOCOL_VERSION = 2  # v1 (patience-8 early stopping, 40-epoch cosine) is superseded, see DECISIONS D-26


def git_state() -> dict:
    """HEAD commit and whether the *code* tree is dirty. results/ and logs/ are ignored here: a
    half-written fold left by an interrupted run must not block a restart."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)results", ":(exclude)logs"],
            text=True).strip())
        return {"commit": sha, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


# Captured ONCE at import: the code a long-running process executes is the code it loaded, so
# results are tagged with the commit (and dirty flag) of process start, not of whenever a fold ends.
GIT_AT_START = git_state()


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def amp_context(device, amp: str):
    if amp == "none":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[amp]
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_scheduler(opt, warmup_steps: int, total_steps: int):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


@torch.no_grad()
def evaluate(model, loader, device, amp, criterion):
    model.eval()
    losses, preds, labels, rows = [], [], [], []
    for x, y, r in loader:
        x, y = x.to(device), y.to(device)
        with amp_context(device, amp):
            logits = model(x)
        losses.append(criterion(logits.float(), y).item() * len(y))
        preds.append(logits.argmax(1).cpu())
        labels.append(y.cpu())
        rows.append(r)
    y_true, y_pred = torch.cat(labels).numpy(), torch.cat(preds).numpy()
    return {"loss": sum(losses) / len(y_true), "y_true": y_true, "y_pred": y_pred,
            "rows": torch.cat(rows).numpy()}


def run_round(cfg: dict, k: int, results_dir="results", ckpt_dir="checkpoints",
              device="auto", store: FeatureStore | None = None, log=print) -> dict:
    exp = cfg["experiment"]
    tcfg, dcfg = cfg["train"], cfg["data"]
    if cfg.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"{exp}: config must declare protocol_version: {PROTOCOL_VERSION}")
    if "early_stopping" not in tcfg:
        raise ValueError(f"{exp}: train.early_stopping must be set explicitly (true/false)")
    if tcfg.get("amp") not in ("none", "fp16", "bf16"):
        raise ValueError(f"{exp}: train.amp must be none|fp16|bf16 here, got {tcfg.get('amp')!r}")
    out_dir = Path(results_dir) / exp
    ck_dir = Path(ckpt_dir) / exp / f"fold{k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"fold{k}.json"
    if result_path.exists():
        log(f"[{exp} fold {k}] result exists, skipping (test fold is evaluated once)")
        return json.loads(result_path.read_text())

    device = pick_device(device)
    seed_everything(tcfg["seed"])
    store = store or FeatureStore(dcfg["cache"])
    is_ast = dcfg.get("input", "logmel") == "ast"
    if is_ast:
        train_ds, val_ds, test_ds, stats = make_round_ast(store, k)
    else:
        train_ds, val_ds, test_ds, stats = make_round(store, k, dcfg["floor_db"])

    model = build_model(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    steps_per_epoch = len(train_ds) // tcfg["batch_size"]  # drop_last
    sched = make_scheduler(opt, tcfg["warmup_epochs"] * steps_per_epoch,
                           tcfg["max_epochs"] * steps_per_epoch)
    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg["label_smoothing"])
    eval_criterion = nn.CrossEntropyLoss()  # plain CE for reporting val loss
    amp = tcfg.get("amp", "none")
    scaler = None
    if amp == "fp16":
        try:
            scaler = torch.amp.GradScaler(device.type)
        except Exception as e:  # not available on this backend
            log(f"  GradScaler unavailable on {device.type} ({e}); fp16 without loss scaling")
    sa_cfg = cfg.get("specaugment", {})
    axes = {"freq_dim": 2, "time_dim": 1} if is_ast else {}  # AST inputs are (B, T, F)
    aug = SpecAugment(**sa_cfg, **axes, generator=torch.Generator())

    state = {"epoch": 0, "best_val_acc": -1.0, "best_epoch": 0, "bad": 0, "history": [],
             "train_time_s": 0.0, "done": False}
    last_p, best_p = ck_dir / "last.pt", ck_dir / "best.pt"
    if last_p.exists():  # resume
        ck = torch.load(last_p, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        state = ck["state"]
        log(f"[{exp} fold {k}] resumed after epoch {state['epoch']}")

    log(f"[{exp} fold {k}] device={device} params={n_params:,} train/val/test="
        f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)} norm mean={stats.mean:.2f} std={stats.std:.2f}")

    val_loader = eval_loader(val_ds, 256)
    jsonl = out_dir / f"fold{k}_epochs.jsonl"
    while not state["done"] and state["epoch"] < tcfg["max_epochs"]:
        ep = state["epoch"] + 1
        t0 = time.time()
        torch.manual_seed(tcfg["seed"] * 1000 + ep)  # dropout etc. reproducible across resume
        aug.generator.manual_seed(tcfg["seed"] * 1000 + ep)
        model.train()
        aug.train()
        loss_sum = correct = seen = 0
        for x, y, _ in train_loader(train_ds, tcfg["batch_size"], tcfg["seed"], ep):
            x, y = aug(x).to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with amp_context(device, amp):
                logits = model(x)
            loss = criterion(logits.float(), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {ep}: {loss.item()}")
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            sched.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        train_time = time.time() - t0
        v = evaluate(model, val_loader, device, amp, eval_criterion)
        val_acc = float((v["y_true"] == v["y_pred"]).mean())
        val_f1 = classification_metrics(v["y_true"], v["y_pred"])["macro_f1"]
        rec = {"epoch": ep, "train_loss": loss_sum / seen, "train_acc": correct / seen,
               "val_loss": v["loss"], "val_acc": val_acc, "val_macro_f1": val_f1,
               "lr": opt.param_groups[0]["lr"], "train_epoch_s": round(train_time, 2),
               "epoch_total_s": round(time.time() - t0, 2)}
        state["history"].append(rec)
        state["epoch"] = ep
        state["train_time_s"] += time.time() - t0
        if val_acc > state["best_val_acc"]:
            state.update(best_val_acc=val_acc, best_epoch=ep, bad=0)
            atomic_save(model.state_dict(), best_p)
            (ck_dir / "best.json").write_text(json.dumps({
                "experiment": exp, "fold": k, "epoch": ep, "val_acc": val_acc, "model": cfg["model"],
                "config_hash": config_hash(cfg), "git": GIT_AT_START, "protocol_version": PROTOCOL_VERSION}))
        else:
            state["bad"] += 1
        if (tcfg["early_stopping"] and state["bad"] >= tcfg["patience"]) or ep >= tcfg["max_epochs"]:
            state["done"] = True
        atomic_save({"model": model.state_dict(), "opt": opt.state_dict(),
                     "sched": sched.state_dict(), "state": state}, last_p)
        with open(jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"  ep {ep:2d} train {rec['train_loss']:.3f}/{rec['train_acc']:.3f} "
            f"val {rec['val_loss']:.3f}/{val_acc:.3f} f1 {val_f1:.3f} "
            f"lr {rec['lr']:.2e} {rec['epoch_total_s']:.1f}s" + (" *" if state["best_epoch"] == ep else ""))

    # Single, final test-fold evaluation with the best-validation checkpoint.
    model.load_state_dict(torch.load(best_p, map_location="cpu"))
    model.to(device)
    t = evaluate(model, eval_loader(test_ds, 256), device, amp, eval_criterion)
    m = classification_metrics(t["y_true"], t["y_pred"])
    m["loss"] = t["loss"]
    m["by_padding"] = subset_accuracy(t["y_true"], t["y_pred"], store.padded[t["rows"]])
    result = {
        "experiment": exp, "fold": k, "protocol_version": PROTOCOL_VERSION,
        "early_stopping": tcfg["early_stopping"], "classes": CLASSES, **m,
        "n_test": int(len(t["y_true"])), "n_params": n_params,
        "best_epoch": state["best_epoch"], "best_val_acc": state["best_val_acc"],
        "epochs_run": state["epoch"], "stopped_early": state["epoch"] < tcfg["max_epochs"],
        "train_time_s": round(state["train_time_s"], 1),
        "mean_epoch_s": round(float(np.mean([h["epoch_total_s"] for h in state["history"]])), 2),
        "norm": {"mean": stats.mean, "std": stats.std, "floor_db": dcfg.get("floor_db")},
        "device": str(device), "amp": amp, "torch": torch.__version__, "git": GIT_AT_START,
        "config_hash": config_hash(cfg), "model_provenance": getattr(model, "provenance", None),
        "config": cfg, "history": state["history"],
    }
    with open(result_path, "x") as f:  # 'x': refuse to overwrite -> test fold touched once
        json.dump(result, f, indent=2)
    import pandas as pd
    pd.DataFrame({"slice_file_name": store.index.slice_file_name.values[t["rows"]],
                  "y_true": t["y_true"], "y_pred": t["y_pred"]}).to_csv(
        out_dir / f"fold{k}_preds.csv", index=False)
    last_p.unlink(missing_ok=True)  # resume file only; best.pt is kept for quantization/pruning/latency
    del model, opt
    if device.type == "mps":
        torch.mps.empty_cache()
    log(f"[{exp} fold {k}] TEST acc {m['accuracy']:.4f} macro-F1 {m['macro_f1']:.4f} "
        f"(best epoch {state['best_epoch']}, {state['train_time_s']:.0f}s train)")
    return result
