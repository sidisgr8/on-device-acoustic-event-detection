"""Parameters and multiply-accumulates (MACs) per clip, counted from the actual model code.

MACs here = Conv2d + Linear + GRU multiply-accumulates for one forward pass of one clip (batch 1). Not counted:
BatchNorm, activations, pooling, softmax, and (for the CNN family) nothing else; for AST the attention score/value
matmuls (2 x layers x tokens^2 x hidden) are added analytically because they are not Linear layers.
CPU only, no timing. Output: results/complexity.json via scripts/compute_complexity.py.
"""
import torch
from torch import nn

AST_TOKENS = 2 + 12 * 39        # cls + distillation + 12 (freq) x 39 (time) patches for 400 frames


def macs_hooks(model: nn.Module):
    """Register hooks that accumulate MACs in the returned dict; returns (totals, handles). Remove handles after use."""
    total = {"conv": 0, "linear": 0, "gru": 0}
    hs = []

    def conv(m, i, o):
        total["conv"] += o.numel() // o.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def lin(m, i, o):
        total["linear"] += o.numel() // o.shape[0] // m.out_features * m.in_features * m.out_features

    def gru(m, i, o):
        T = i[0].shape[1]
        d = 2 if m.bidirectional else 1
        total["gru"] += T * d * 3 * (m.input_size * m.hidden_size + m.hidden_size * m.hidden_size)

    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            hs.append(mod.register_forward_hook(conv))
        elif isinstance(mod, nn.Linear):
            hs.append(mod.register_forward_hook(lin))
        elif isinstance(mod, nn.GRU):
            hs.append(mod.register_forward_hook(gru))
    return total, hs


def attention_macs(num_layers: int, tokens: int, hidden: int) -> int:
    """QK^T and attention-times-V per layer: 2 x tokens^2 x hidden."""
    return num_layers * 2 * tokens * tokens * hidden


def count(model: nn.Module, x: torch.Tensor) -> dict:
    total, hs = macs_hooks(model)
    with torch.no_grad():
        model.eval()(x)
    for h in hs:
        h.remove()
    return total


def model_complexity(experiment: str) -> dict:
    """Build the architecture of `experiment` (random weights, no download) and count params and MACs."""
    from src.compress.common import find_config
    from src.models import build_model
    cfg = find_config(experiment)
    mc = dict(cfg["model"])
    if "pretrained" in mc:
        mc["pretrained"] = False
    model = build_model(mc)
    ast = cfg["data"].get("input") == "ast"
    x = torch.randn(1, 400, 128) if ast else torch.randn(1, 1, 128, 201)
    parts = count(model, x)
    attn = 0
    if ast:
        c = model.model.config
        attn = attention_macs(c.num_hidden_layers, AST_TOKENS, c.hidden_size)
    total = sum(parts.values()) + attn
    return {"experiment": experiment, "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "macs_conv": parts["conv"], "macs_linear": parts["linear"], "macs_gru": parts["gru"],
            "macs_attention": attn, "macs_total": total, "input_shape": list(x.shape[1:])}
