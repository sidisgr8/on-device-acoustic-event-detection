"""Global magnitude pruning + 3 epochs of masked fine-tuning, evaluated on folds 1, 5, 9 (spec 4.6, D-10, D-31).

    python scripts/prune.py                                  # all models, default sparsities, folds 1 5 9, GPU
    python scripts/prune.py --experiments ast --sparsities 0.3 0.7 --folds 1
    python scripts/prune.py --dry-run --experiments cnn_baseline --folds 1 --out /tmp/p   # CPU, tiny, validation data only

Sparsities: 30/50/70 % for the CNN family, 30/70 % for AST. A finished (experiment, sparsity, fold) is
skipped. Fine-tuning uses the TRAINING folds; model selection among the 3 epochs uses the validation fold;
the test fold is evaluated once at the end (never in --dry-run, which uses validation data only).
"""
import argparse
import copy
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.compress.common import (CKPT_DIR, RESULTS_DIR, available_folds, best_ckpt, find_config,
                                 load_trained_model, model_kind, sha256)
from src.compress.prune import (NOTE, apply_masks, finetune_lr, global_magnitude_masks, sizes,
                                sparsity_report)
from src.data.dataset import (FeatureStore, SpecAugment, eval_loader, make_round, make_round_ast,
                              train_loader)
from src.eval.metrics import classification_metrics, subset_accuracy
from src.train.loop import (GIT_AT_START, amp_context, evaluate, make_scheduler, pick_device,
                            seed_everything)

EXPERIMENTS = ["cnn_baseline", "mobilenetv3_small_scratch", "mobilenetv3_small_imagenet", "crnn", "ast"]
DEFAULT_FOLDS = [1, 5, 9]
EPOCHS = 3


def sparsities_for(exp):
    return [0.3, 0.7] if exp == "ast" else [0.3, 0.5, 0.7]


def val_acc(model, ds, device, amp, limit=None):
    if limit:
        ds = type(ds)(ds.x[:limit], ds.y[:limit], ds.rows[:limit].numpy())
    ev = evaluate(model, eval_loader(ds, 64), device, amp, nn.CrossEntropyLoss())
    return float((ev["y_true"] == ev["y_pred"]).mean()), ev


def run_one(exp, cfg, sparsity, fold, store, device, a, source_cfg):
    tcfg = cfg["train"]
    amp = source_cfg["train"]["amp"] if source_cfg else tcfg["amp"]
    if amp not in ("none", "fp16", "bf16"):
        amp = "none"
    seed_everything(tcfg["seed"])
    ckpt = best_ckpt(exp, fold, a.ckpt_dir)
    model = load_trained_model(cfg, ckpt)
    model = model.to(device)
    if cfg["data"].get("input", "logmel") == "ast":
        tr, va, te, stats = make_round_ast(store, fold)
    else:
        tr, va, te, stats = make_round(store, fold, cfg["data"]["floor_db"])
    base_size = sizes(model)
    base_report = sparsity_report(model)

    masks = global_magnitude_masks(model, sparsity)
    apply_masks(model, masks)
    report0 = sparsity_report(model)
    ft_ds = va if a.dry_run else tr          # dry run: "training" data comes from the validation fold
    max_batches = a.max_batches if a.dry_run else None
    lim = 64 if a.dry_run else None
    acc_before, _ = val_acc(model, va, device, amp, lim)

    epochs = a.ft_epochs or EPOCHS
    lr = a.ft_lr if a.ft_lr else finetune_lr(tcfg["lr"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tcfg["weight_decay"])
    n_steps = max(1, len(ft_ds) // tcfg["batch_size"])
    if max_batches:
        n_steps = min(n_steps, max_batches)
    sched = make_scheduler(opt, 0, epochs * n_steps)       # no warm-up, cosine over the fine-tune epochs
    crit = nn.CrossEntropyLoss(label_smoothing=tcfg["label_smoothing"])
    axes = {"freq_dim": 2, "time_dim": 1} if cfg["data"].get("input") == "ast" else {}
    aug = SpecAugment(**cfg.get("specaugment", {}), **axes, generator=torch.Generator())
    scaler = torch.amp.GradScaler(device.type) if amp == "fp16" else None
    best = {"val_acc": -1.0, "epoch": 0, "state": None}
    history = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        torch.manual_seed(tcfg["seed"] * 1000 + ep)
        aug.generator.manual_seed(tcfg["seed"] * 1000 + ep)
        model.train()
        aug.train()
        for i, (x, y, _) in enumerate(train_loader(ft_ds, tcfg["batch_size"], tcfg["seed"], ep)):
            if i >= n_steps:
                break
            x, y = aug(x).to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with amp_context(device, amp):
                logits = model(x)
            loss = crit(logits.float(), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss {loss.item()} (epoch {ep})")
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            sched.step()
            apply_masks(model, masks)                       # pruned weights stay exactly zero
        acc, _ = val_acc(model, va, device, amp, lim)
        history.append({"epoch": ep, "val_acc": acc})
        if acc > best["val_acc"]:
            best.update(val_acc=acc, epoch=ep, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best["state"])
    apply_masks(model, masks)
    report = sparsity_report(model)
    assert abs(report["actual_sparsity_of_prunable"] - sparsity) < 1e-3, report
    rec = {"experiment": exp, "kind": model_kind(cfg), "fold": fold, "sparsity": sparsity,
           "protocol_version": 2, "recipe": "extended" if a.ft_lr else "main", "dry_run": bool(a.dry_run), "split_evaluated": "val" if a.dry_run else "test",
           "finetune": {"epochs": epochs, "lr": lr,
                        "lr_rule": ("absolute lr, extended fine-tune recipe (D-45, D-47)" if a.ft_lr else "0.1 x original training lr (D-31)"),
                        "batch_size": tcfg["batch_size"], "steps_per_epoch": n_steps, "amp": amp,
                        "best_epoch_by_val": best["epoch"], "history": history,
                        "data": "validation fold (dry run)" if a.dry_run else "training folds"},
           "val_acc_after_pruning_before_finetune": acc_before, "val_acc_best": best["val_acc"],
           **report, "nonzero_params_before": base_report["nonzero_params"],
           "size_dense_fp32_bytes": sizes(model)["dense_fp32_bytes"], "size_gzip_bytes": sizes(model)["gzip_bytes"],
           "size_gzip_bytes_unpruned_baseline": base_size["gzip_bytes"],
           "size_dense_fp32_bytes_unpruned_baseline": base_size["dense_fp32_bytes"],
           "source_best_pt_sha256": sha256(ckpt), "latency_note": NOTE, "torch": torch.__version__,
           "device": str(device), "git": GIT_AT_START, "date": datetime.now().isoformat(timespec="seconds"),
           "train_time_s": round(time.time() - t0, 1)}
    if not a.dry_run:                                       # the one and only test-fold evaluation
        ev = evaluate(model, eval_loader(te, 64), device, amp, nn.CrossEntropyLoss())
        m = classification_metrics(ev["y_true"], ev["y_pred"])
        rec.update(accuracy=m["accuracy"], macro_f1=m["macro_f1"], n_test=int(len(ev["y_true"])),
                   by_padding=subset_accuracy(ev["y_true"], ev["y_pred"], store.padded[ev["rows"]]))
    else:
        _, ev = val_acc(model, va, device, amp, lim)
        m = classification_metrics(ev["y_true"], ev["y_pred"])
        rec.update(val_accuracy=m["accuracy"], val_macro_f1=m["macro_f1"], n_eval=int(len(ev["y_true"])))
    return rec, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", default=EXPERIMENTS)
    ap.add_argument("--folds", nargs="+", type=int, default=DEFAULT_FOLDS)
    ap.add_argument("--sparsities", nargs="+", type=float, default=None)
    ap.add_argument("--ft-lr", type=float, default=None, help="absolute fine-tune lr (extended recipe, D-47); default 0.1 x training lr")
    ap.add_argument("--ft-epochs", type=int, default=None, help="fine-tune epochs (default 3)")
    ap.add_argument("--dir-suffix", default="", help="suffix of the sparsity directory, e.g. _ext")
    ap.add_argument("--out", default=str(RESULTS_DIR / "prune"))
    ap.add_argument("--artifacts", default=str(CKPT_DIR / "pruned"))
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU, a few batches of VALIDATION data as 'training' data, validation-only evaluation")
    ap.add_argument("--max-batches", type=int, default=3)
    a = ap.parse_args()
    device = torch.device("cpu") if a.dry_run else pick_device(a.device)
    store = FeatureStore()
    for exp in a.experiments:
        cfg = find_config(exp)
        done = set(available_folds(exp, a.ckpt_dir, a.results_dir))
        for fold in a.folds:
            if fold not in done and not (a.dry_run and best_ckpt(exp, fold, a.ckpt_dir).exists()):
                print(f"[{exp} fold {fold}] no finished fold / best.pt yet, skipping")
                continue
            rj = Path(a.results_dir) / exp / f"fold{fold}.json"
            source_cfg = json.loads(rj.read_text())["config"] if rj.exists() else None
            for sp in (a.sparsities or sparsities_for(exp)):
                path = Path(a.out) / exp / f"s{int(round(sp * 100))}{a.dir_suffix}" / f"fold{fold}.json"
                if path.exists() and not a.dry_run:
                    continue
                print(f"[{exp} fold {fold}] sparsity {sp:.0%} on {device}", flush=True)
                try:
                    rec, model = run_one(exp, cfg, sp, fold, store, device, a, source_cfg)
                except Exception as e:  # record and move on
                    print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    (path.parent / f"fold{fold}.error.txt").write_text(f"{type(e).__name__}: {e}\n")
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w" if a.dry_run else "x") as f:
                    json.dump(rec, f, indent=2)
                ck = Path(a.artifacts) / exp / path.parent.name / f"fold{fold}"
                ck.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ck / "pruned.pt")
                key = "val_accuracy" if a.dry_run else "accuracy"
                print(f"  nonzero {rec['nonzero_params']:,}/{rec['total_params']:,} "
                      f"gzip {rec['size_gzip_bytes'] / 1e6:.2f} MB (unpruned {rec['size_gzip_bytes_unpruned_baseline'] / 1e6:.2f}) "
                      f"val-before-ft {rec['val_acc_after_pruning_before_finetune']:.3f} -> best val {rec['val_acc_best']:.3f} "
                      f"{key}={rec[key]:.3f}", flush=True)


if __name__ == "__main__":
    main()
