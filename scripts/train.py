"""Train one or more cross-validation rounds of an experiment.

    python scripts/train.py --config configs/cnn_baseline.yaml --folds 1 5 9
"""
import argparse

import yaml

from src.data.dataset import FeatureStore
from src.train.loop import run_round


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config entries, e.g. train.max_epochs=3 experiment=smoke")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    for kv in a.set:
        key, val = kv.split("=", 1)
        node = cfg
        *path, last = key.split(".")
        for p in path:
            node = node[p]
        node[last] = yaml.safe_load(val)
    store = FeatureStore(cfg["data"]["cache"])
    for k in a.folds:
        run_round(cfg, k, a.results_dir, a.ckpt_dir, a.device, store)


# Guard keeps the module importable and safe with multiprocessing spawn.
if __name__ == "__main__":
    main()
