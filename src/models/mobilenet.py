"""Model 1b/1c: MobileNetV3-Small (timm `mobilenetv3_small_100`), 1-channel input, 10-way head."""
import hashlib
from pathlib import Path

import timm
from torch import nn

ARCH = "mobilenetv3_small_100"


def weights_provenance(model) -> dict:
    """Which pretrained weights were loaded and where they came from (for the results JSON)."""
    cfg = model.pretrained_cfg
    info = {"timm_version": timm.__version__, "arch": ARCH, "pretrained_tag": cfg.get("tag"),
            "hf_hub_id": cfg.get("hf_hub_id"), "source_url": cfg.get("url") or cfg.get("hf_hub_id")}
    try:
        from huggingface_hub import hf_hub_download
        f = Path(hf_hub_download(cfg["hf_hub_id"], "model.safetensors"))
        info["file"] = f.name
        info["sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
        info["hf_snapshot"] = f.parent.name  # commit hash of the HF repo revision (snapshots/<hash>/)
    except Exception as e:  # provenance is best-effort, never blocks training
        info["sha256_error"] = repr(e)
    return info


class MobileNetV3Small(nn.Module):
    """timm builds the 1-channel stem by SUMMING the pretrained RGB filters (adapt_input_conv),
    which is what the spec asks for; with pretrained=False the stem is simply 1-channel."""

    def __init__(self, pretrained=False, n_classes=10):
        super().__init__()
        self.net = timm.create_model(ARCH, pretrained=pretrained, in_chans=1, num_classes=n_classes)
        self.provenance = weights_provenance(self.net) if pretrained else {"pretrained": False}

    def forward(self, x):
        return self.net(x)
