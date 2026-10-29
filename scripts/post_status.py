"""Progress of the post-training run (scripts/launch_post.sh): which stage, how many results exist, failures.

    .venv-post/bin/python scripts/post_status.py
"""
import json
import os
import subprocess
from pathlib import Path

R = Path("results")
A1 = "a1_mobilenetv3_small_scratch_lr3e4"
MODELS = ["cnn_baseline", "mobilenetv3_small_scratch", "mobilenetv3_small_imagenet", "crnn", "ast"]
PRUNE_FOLDS = 3


def alive():
    p = Path("logs/post/run_post.pid")
    if not p.exists():
        return None
    pid = int(p.read_text())
    stat = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return pid if stat and not stat.startswith("Z") else None


def count_json(root, pred=lambda r: True):
    n = 0
    for f in Path(root).rglob("fold*.json"):
        try:
            if pred(json.loads(f.read_text())):
                n += 1
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    return n


def main():
    pid = alive()
    print(f"post run: {'RUNNING pid ' + str(pid) if pid else 'NOT RUNNING'}")
    out = Path("logs/post/run1.out")
    if out.exists():
        for line in out.read_text().splitlines()[-6:]:
            print("  " + line)
    a1 = len(list((R / A1).glob("fold[0-9]*.json"))) if (R / A1).exists() else 0
    exp_prune = sum(PRUNE_FOLDS * (2 if m == "ast" else 3) for m in MODELS)
    pr = count_json(R / "prune", lambda r: not r.get("dry_run") and r.get("split", r.get("split_evaluated")) == "test")
    exp_quant = len(MODELS) * 10 * 4
    qu = count_json(R / "quant", lambda r: r.get("split") == "test")
    errs = [str(p) for d in ("prune", "quant") for p in (R / d).rglob("*.error.txt")]
    print(f"\nA-1 ablation folds   {a1:>3}/10")
    print(f"pruning results      {pr:>3}/{exp_prune}")
    print(f"quantization results {qu:>3}/{exp_quant}  (variant x fold files)")
    fail = Path("logs/post/failures.jsonl")
    print(f"failed stages: {len(fail.read_text().splitlines()) if fail.exists() else 0}   failed items (error.txt): {len(errs)}")
    for e in errs:
        print("  " + e)
    qf = Path("logs/failures.jsonl")
    if qf.exists():
        print(f"training failures (logs/failures.jsonl): {len(qf.read_text().splitlines())}")
    print("latency: NOT part of this run (manual, idle machine)")


if __name__ == "__main__":
    main()
