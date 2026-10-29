"""Prove that a change did not alter anything the CNN-family training depends on.

Compares the working tree against a baseline commit (default: the commit the running queue captured at start,
logs/queue_state.json -> git.commit) on three independent points:

 1. config hashes of the four CNN-family configs: baseline commit vs working tree vs the `config_hash` already
    recorded in finished result files (the hash is of the exact config dict a fold was trained with);
 2. SpecAugment output, bit for bit (sha256 of the tensor bytes), for the CNN-family settings and several seeds,
    produced by the BASELINE source tree (extracted with git archive) and by the working tree, in separate processes;
 3. the list of files under the training-relevant source directories that differ from the baseline.

    python scripts/verify_cnn_family_unchanged.py [--baseline COMMIT]
Exit code 0 only if everything is identical.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

CNN_FAMILY = ["cnn_baseline", "mobilenetv3_small_scratch", "mobilenetv3_small_imagenet", "crnn"]
TRAINING_PATHS = ["src/train", "src/data", "src/models", "src/features", "src/eval"]
DIGEST_CODE = r"""
import hashlib, json, sys, yaml, torch
from src.data import dataset
assert sys.argv[1] in dataset.__file__, (sys.argv[1], dataset.__file__)   # really imported from that tree
cfg = yaml.safe_load(open(sys.argv[2]))["specaugment"]
out = {}
for seed in (0, 1, 7, 42, 4242):
    g = torch.Generator().manual_seed(seed)
    aug = dataset.SpecAugment(**cfg, generator=g).train()
    x = torch.randn(64, 1, 128, 201, generator=torch.Generator().manual_seed(seed + 1))
    y = aug(x)
    out[seed] = hashlib.sha256(y.numpy().tobytes()).hexdigest()[:16]
print(json.dumps(out))
"""


def sh(*a, cwd=None, env=None):
    return subprocess.run(a, capture_output=True, text=True, check=True, cwd=cwd, env=env).stdout


def config_hash(cfg: dict) -> str:  # same function as src.train.loop.config_hash (kept standalone: no heavy imports)
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def digests(tree: Path, cfg_path: Path) -> dict:
    env = {"PYTHONPATH": str(tree), "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1"}
    out = sh(sys.executable, "-c", DIGEST_CODE, str(tree), str(cfg_path), cwd=tree, env=env)
    return json.loads(out.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=None)
    a = ap.parse_args()
    base = a.baseline or json.loads(Path("logs/queue_state.json").read_text())["git"]["commit"]
    print(f"baseline commit: {base}\n")
    ok = True

    print("1. config hashes (baseline commit | working tree | recorded in finished result files)")
    for exp in CNN_FAMILY:
        path = f"configs/{exp}.yaml"
        h_base = config_hash(yaml.safe_load(sh("git", "show", f"{base}:{path}")))
        h_now = config_hash(yaml.safe_load(open(path)))
        recorded = sorted({json.loads(p.read_text())["config_hash"] for p in Path("results", exp).glob("fold*.json")
                           if json.loads(p.read_text()).get("protocol_version") == 2})
        same = h_base == h_now and all(r == h_now for r in recorded)
        ok &= same
        print(f"   {exp:28s} {h_base} | {h_now} | {recorded or 'no finished folds yet'}  {'IDENTICAL' if same else 'DIFFERENT !!'}")

    print("\n2. SpecAugment output, bit for bit, CNN-family settings (baseline source tree vs working tree)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        arch = subprocess.run(["git", "archive", base, "src", "configs"], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(tmp)], input=arch, check=True)
        for exp in CNN_FAMILY:
            d_base = digests(tmp, tmp / "configs" / f"{exp}.yaml")
            d_now = digests(Path.cwd(), Path(f"configs/{exp}.yaml"))
            same = d_base == d_now
            ok &= same
            print(f"   {exp:28s} {len(d_now)} seeds, first digest {d_now['0']}  {'BIT-IDENTICAL' if same else 'DIFFERENT !!'}")

    print("\n3. files that differ from the baseline in the training-relevant source directories")
    changed = [l for l in sh("git", "diff", "--name-only", base, "--", *TRAINING_PATHS).splitlines() if l]
    print("   " + ("\n   ".join(changed) if changed else "none"))
    ok &= not changed
    print("\nCNN-family: " + ("UNCHANGED" if ok else "CHANGED - do not proceed"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
