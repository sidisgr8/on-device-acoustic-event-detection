"""Print layer-by-layer tensor shapes, parameter counts and multiply-accumulates of every model, from the
actual code and configs (used to write docs/study/METHODS_FACTSHEET.md). CPU only, tiny inputs, no timing.

    python scripts/describe_models.py > docs/study/generated/model_summaries.md
"""
import warnings

import torch
from torch import nn

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

from src.analysis.complexity import macs_hooks
from src.compress.common import find_config
from src.models import build_model


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def trace(model, x, modules):
    """Run once; return rows (name, type, out_shape, own_params) for the listed (name, module) pairs."""
    rows, hs = [], []
    for name, mod in modules:
        def hook(m, i, o, name=name):
            o = o[0] if isinstance(o, tuple) else o
            rows.append((name, type(m).__name__, tuple(o.shape), n_params(m)))
        hs.append(mod.register_forward_hook(hook))
    total, mh = macs_hooks(model)
    with torch.no_grad():
        model.eval()(x)
    for h in hs + mh:
        h.remove()
    return rows, total


def md(rows, header=("layer", "type", "output shape (B=1)", "params")):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(f"{c:,}" if isinstance(c, int) else str(c) for c in r) + " |")
    return "\n".join(out)


def section(title, rows, total, params, extra=""):
    macs = sum(total.values())
    print(f"\n## {title}\n")
    print(md(rows))
    print(f"\n**Total parameters: {params:,}** (trainable). Conv/Linear/GRU multiply-accumulates per clip: "
          f"{macs / 1e6:,.1f} M (conv {total['conv'] / 1e6:,.1f} M, linear {total['linear'] / 1e6:,.1f} M, "
          f"GRU {total['gru'] / 1e6:,.1f} M).{extra}")


def main():
    print("# Generated model summaries (scripts/describe_models.py; do not edit by hand)\n")
    print("Shapes are for one clip (batch 1). MAC counts include only Conv2d, Linear and GRU layers "
          "(no BatchNorm, activations, pooling, softmax; for AST the attention matmuls are added separately).")
    x = torch.randn(1, 1, 128, 201)

    m = build_model(find_config("cnn_baseline")["model"])
    mods = [(f"features.{i}", b) for i, b in enumerate(m.features)] + [("head.1 (Linear)", m.head[1])]
    rows, total = trace(m, x, mods)
    rows.insert(len(rows) - 1, ("global average pool (mean over F,T)", "ops", (1, 256), 0))
    section("CNN baseline (4 x [conv3x3-BN-ReLU-maxpool2], GAP, dropout 0.3, linear)", rows, total, n_params(m))
    print("\nPer block: Conv2d(no bias) -> BatchNorm2d -> ReLU -> MaxPool2d(2). Channels 1->32->64->128->256.")
    for i, b in enumerate(m.features):
        print(f"- features.{i}: conv {n_params(b[0]):,} + BN {n_params(b[1]):,} = {n_params(b):,} params")

    for exp in ("mobilenetv3_small_scratch",):
        cfg = find_config(exp)
        mc = dict(cfg["model"]); mc["pretrained"] = False
        m = build_model(mc)
        net = m.net
        mods = [("conv_stem", net.conv_stem), ("bn1 (BN+hardswish)", net.bn1)]
        mods += [(f"blocks.{i}", b) for i, b in enumerate(net.blocks)]
        mods += [("global_pool", net.global_pool), ("conv_head (1x1)", net.conv_head), ("classifier", net.classifier)]
        rows, total = trace(m, x, mods)
        section("MobileNetV3-Small (timm `mobilenetv3_small_100`, 1-channel stem, 10-way classifier)", rows, total, n_params(m))
        print("\nBlocks are inverted-residual (depthwise-separable) stages; several use squeeze-excite and hard-swish. "
              f"Stage module counts: {[len(b) for b in net.blocks]}. Head: 1x1 conv to 1024 channels, hard-swish, linear 10. "
              f"Head dropout (timm drop_rate): {net.drop_rate}.")
        print(f"Stem weight shape (out,in,kh,kw) = {tuple(net.conv_stem.weight.shape)} (1 input channel); "
              "ImageNet run: timm sums the pretrained RGB filters into this single-channel filter.")

    m = build_model(find_config("crnn")["model"])
    mods = [(f"convs.{i}", b) for i, b in enumerate(m.convs)] + [("proj (Linear 2048->128)", m.proj),
                                                                    ("gru (BiGRU, hidden 128)", m.gru),
                                                                    ("head.1 (Linear 512->10)", m.head[1])]
    rows, total = trace(m, x, mods)
    _, shapes = m(x, return_shapes=True)
    section("CRNN (3 conv blocks -> linear 128 -> BiGRU 128 -> [mean;max] over time -> dropout 0.3 -> linear)", rows, total, n_params(m))
    print("\nShapes through the network (from `CRNN.forward(return_shapes=True)`):\n")
    for k, v in shapes.items():
        print(f"- {k}: {v}")
    print(f"\nGRU parameters: {n_params(m.gru):,} (2 directions x 3 gates x (128x128 input + 128x128 recurrent + 2 biases)).")

    cfg = find_config("ast")
    mc = dict(cfg["model"]); mc["pretrained"] = False
    m = build_model(mc)
    a = m.model.audio_spectrogram_transformer
    xa = torch.randn(1, 400, 128)
    mods = [("embeddings (patch + cls/dist + pos)", a.embeddings),
            ("embeddings.patch_embeddings.projection (Conv2d 16x16, stride 10)", a.embeddings.patch_embeddings.projection),
            ("layers.0 (transformer block)", a.layers[0]), ("layers.11 (transformer block)", a.layers[11]),
            ("layernorm", a.layernorm), ("classifier.layernorm", m.model.classifier.layernorm),
            ("classifier.dense (Linear 768->10)", m.model.classifier.dense)]
    rows, total = trace(m, xa, mods)
    section("AST (HF `ASTForAudioClassification`, 400-frame input, AudioSet checkpoint architecture)", rows, total, n_params(m))
    cfgm = m.model.config
    T = 2 + 12 * 39
    attn = cfgm.num_hidden_layers * 2 * T * T * cfgm.hidden_size
    print(f"\nTokens: 2 special + 12 (freq) x 39 (time) = {T}. Attention score+value matmul MACs (analytic, 12 layers x 2 x T^2 x D): "
          f"{attn / 1e6:,.1f} M; so total ~ {(sum(total.values()) + attn) / 1e6:,.1f} M MACs per clip.")
    print(f"Config: hidden {cfgm.hidden_size}, layers {cfgm.num_hidden_layers}, heads {cfgm.num_attention_heads}, "
          f"MLP {cfgm.intermediate_size}, patch {cfgm.patch_size}, strides (freq {cfgm.frequency_stride}, time {cfgm.time_stride}), "
          f"max_length {cfgm.max_length}, mel bins {cfgm.num_mel_bins}, hidden dropout {cfgm.hidden_dropout_prob}, "
          f"attention dropout {cfgm.attention_probs_dropout_prob}.")
    emb = a.embeddings
    print(f"Parameter groups: patch conv {n_params(emb.patch_embeddings):,}; position embeddings {emb.position_embeddings.numel():,} "
          f"{tuple(emb.position_embeddings.shape)}; cls+distillation tokens {emb.cls_token.numel() + emb.distillation_token.numel():,}; "
          f"one transformer block {n_params(a.layers[0]):,} (x12 = {12 * n_params(a.layers[0]):,}); final layernorm {n_params(a.layernorm):,}; "
          f"classifier {n_params(m.model.classifier):,} (LayerNorm {n_params(m.model.classifier.layernorm):,} + dense {n_params(m.model.classifier.dense):,}).")


if __name__ == "__main__":
    main()
