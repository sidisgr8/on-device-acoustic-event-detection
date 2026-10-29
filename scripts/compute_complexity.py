"""Count parameters and MACs per clip for every model from the model code (CPU, no timing).

    python scripts/compute_complexity.py            # writes results/complexity.json (tracked, small)
"""
import json
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

from src.analysis.complexity import model_complexity

EXPERIMENTS = ["cnn_baseline", "mobilenetv3_small_scratch", "mobilenetv3_small_imagenet", "crnn", "ast"]


def main():
    models = {e: model_complexity(e) for e in EXPERIMENTS}
    out = {"note": ("MACs per 4 s clip (batch 1), counted by forward hooks on Conv2d, Linear and GRU layers; "
                    "BatchNorm, activations, pooling and softmax are not counted; AST attention score/value matmuls "
                    "are added analytically (2 x layers x tokens^2 x hidden). Architecture only, pretrained weights "
                    "do not change it (the two MobileNetV3 runs are identical)."),
           "torch": torch.__version__, "models": models}
    Path("results/complexity.json").write_text(json.dumps(out, indent=2))
    for e, m in models.items():
        print(f"{e:28s} params {m['params']:>11,}  MACs {m['macs_total'] / 1e6:>10,.1f} M")


if __name__ == "__main__":
    main()
