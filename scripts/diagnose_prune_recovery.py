"""Is the ImageNet MobileNetV3-S pruning collapse a recovery problem? (VALIDATION data only, D-20)

For folds 1, 5, 9 and sparsity 50 / 70 %: prune exactly as in scripts/prune.py, then fine-tune on the TRAINING folds with
other learning rates / numbers of epochs than the main run (0.1 x original lr = 3e-5 for 3 epochs), and record the
VALIDATION accuracy only. The test fold is never loaded into a metric. Results: results/diagnostics/prune_recovery_val.json

    .venv-post/bin/python scripts/diagnose_prune_recovery.py [--experiments mobilenetv3_small_imagenet ...]
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from scripts.prune import val_acc
from src.compress.common import best_ckpt, find_config, load_trained_model
from src.compress.prune import apply_masks, global_magnitude_masks, sparsity_report
from src.data.dataset import FeatureStore, SpecAugment, make_round, train_loader
from src.train.loop import make_scheduler, pick_device, seed_everything

CONFIGS = [(1e-4, 3), (3e-4, 3), (3e-4, 10)]      # (fine-tune lr, epochs); main run: (0.1 x lr, 3)


def finetune(model, masks, tr, va, cfg, lr, epochs, device):
    t = cfg["train"]
    seed_everything(t["seed"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=t["weight_decay"])
    n_steps = len(tr) // t["batch_size"]
    sched = make_scheduler(opt, 0, epochs * n_steps)
    crit = nn.CrossEntropyLoss(label_smoothing=t["label_smoothing"])
    aug = SpecAugment(**cfg.get("specaugment", {}), generator=torch.Generator())
    hist = []
    for ep in range(1, epochs + 1):
        torch.manual_seed(t["seed"] * 1000 + ep)
        aug.generator.manual_seed(t["seed"] * 1000 + ep)
        model.train()
        aug.train()
        for i, (x, y, _) in enumerate(train_loader(tr, t["batch_size"], t["seed"], ep)):
            if i >= n_steps:
                break
            x, y = aug(x).to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x).float(), y)
            loss.backward()
            opt.step()
            sched.step()
            apply_masks(model, masks)
        acc, _ = val_acc(model, va, device, "none")
        hist.append(acc)
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", default=["mobilenetv3_small_imagenet"])
    ap.add_argument("--out", default="results/diagnostics/prune_recovery_val.json")
    a = ap.parse_args()
    device = pick_device("auto")
    store = FeatureStore()
    out = {"split": "validation only (D-20)", "configs": CONFIGS, "runs": []}
    for exp in a.experiments:
        cfg = find_config(exp)
        for fold in (1, 5, 9):
            tr, va, _, _ = make_round(store, fold, cfg["data"]["floor_db"])
            ref = {sp: json.loads(Path(f"results/prune/{exp}/s{int(sp * 100)}/fold{fold}.json").read_text()) for sp in (0.5, 0.7)}
            for sp in (0.5, 0.7):
                r0 = ref[sp]
                out["runs"].append({"experiment": exp, "fold": fold, "sparsity": sp, "lr": r0["finetune"]["lr"], "epochs": 3,
                                    "main_run_reference": True, "val_acc_before_finetune": r0["val_acc_after_pruning_before_finetune"],
                                    "val_acc_best": r0["val_acc_best"]})
                for lr, epochs in CONFIGS:
                    model = load_trained_model(cfg, best_ckpt(exp, fold)).to(device)
                    masks = global_magnitude_masks(model, sp)
                    apply_masks(model, masks)
                    before, _ = val_acc(model, va, device, "none")
                    t0 = time.time()
                    hist = finetune(model, masks, tr, va, cfg, lr, epochs, device)
                    assert abs(sparsity_report(model)["actual_sparsity_of_prunable"] - sp) < 1e-3
                    out["runs"].append({"experiment": exp, "fold": fold, "sparsity": sp, "lr": lr, "epochs": epochs,
                                        "val_acc_before_finetune": before, "val_acc_by_epoch": hist, "val_acc_best": max(hist),
                                        "seconds": round(time.time() - t0, 1)})
                    print(exp, fold, sp, lr, epochs, f"before {before:.3f} best {max(hist):.3f}", flush=True)
                    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
