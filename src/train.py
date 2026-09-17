"""Train and evaluate one model over chosen cross-validation folds.

Example (quick preliminary run, 3 folds, 8 epochs):
    python -m src.train --model cnn --folds 1,5,9 --epochs 8 --patience 3 --device mps

Per fold: train on 8 folds, pick the best epoch on the validation fold, then evaluate
that checkpoint on the test fold exactly once. Results -> results/<model>/fold<k>.json
"""
import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .data import CLASSES, ClipDataset, fold_split, load_cache, train_stats
from .models import MODELS, build_model


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "no-git"


def pick_device(name):
    if name == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but not available; use --device cpu or cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    return torch.device(name)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu()); labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def lr_factor(epoch, total, warmup=2):
    """2 warm-up epochs, then cosine decay to zero."""
    if epoch < warmup:
        return (epoch + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, total - warmup)))


def run_fold(args, k, X, y, fold, device):
    set_seed(args.seed)
    tr_m, va_m, te_m = fold_split(fold, k)
    mean, std = train_stats(X, tr_m)                      # training folds only
    mk = lambda m, aug: DataLoader(
        ClipDataset(X, y, np.where(m)[0], mean, std, augment=aug, seed=args.seed),
        batch_size=args.batch_size, shuffle=aug, num_workers=args.workers)
    train_dl, val_dl, test_dl = mk(tr_m, True), mk(va_m, False), mk(te_m, False)

    builder, lr, _ = MODELS[args.model]
    model = builder().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: lr_factor(e, args.epochs))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc, best_state, bad, curve = -1.0, None, 0, []
    for epoch in range(args.epochs):
        model.train(); t0 = time.time(); running = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
            running += loss.item() * len(yb)
        sched.step()
        p, l = predict(model, val_dl, device)
        val_acc = accuracy_score(l, p)
        curve.append({"epoch": epoch + 1, "train_loss": running / len(train_dl.dataset),
                      "val_acc": val_acc, "seconds": round(time.time() - t0, 1)})
        print(f"fold {k} epoch {epoch+1}/{args.epochs} loss {curve[-1]['train_loss']:.3f} "
              f"val_acc {val_acc:.3f} ({curve[-1]['seconds']}s)", flush=True)
        if val_acc > best_acc:
            best_acc, bad = val_acc, 0
            best_state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping"); break

    model.load_state_dict(best_state)                      # best-validation checkpoint
    p, l = predict(model, test_dl, device)                 # test fold touched ONCE, here
    result = {
        "model": args.model, "fold": k, "test_accuracy": float(accuracy_score(l, p)),
        "test_macro_f1": float(f1_score(l, p, average="macro")), "best_val_accuracy": best_acc,
        "epochs_run": len(curve), "epoch_cap": args.epochs, "patience": args.patience,
        "confusion_matrix": confusion_matrix(l, p, labels=list(range(10))).tolist(),
        "classes": CLASSES, "curve": curve, "seed": args.seed, "device": str(device),
        "git_hash": git_hash(), "n_train": int(tr_m.sum()), "n_val": int(va_m.sum()),
        "n_test": int(te_m.sum()),
    }
    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)
    (out / f"fold{k}.json").write_text(json.dumps(result, indent=1))
    print(f"fold {k}: test acc {result['test_accuracy']:.4f}  macro-F1 {result['test_macro_f1']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--folds", default="1,5,9", help="comma list of test folds, e.g. 1,5,9 or 1,2,...,10")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu", help="mps | cuda | cpu")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y, fold = load_cache(args.cache)
    device = pick_device(args.device)
    for k in [int(f) for f in args.folds.split(",")]:
        run_fold(args, k, X, y, fold, device)


if __name__ == "__main__":
    main()
