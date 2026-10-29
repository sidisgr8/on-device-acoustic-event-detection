import pytest
import torch
from torch import nn

from src.compress.prune import (apply_masks, finetune_lr, global_magnitude_masks, prunable_tensors, sizes,
                                sparsity_report)
from src.models import build_model


def test_global_sparsity_is_exact_and_spans_conv_linear_gru():
    torch.manual_seed(0)
    m = build_model({"name": "crnn"})
    names = set(prunable_tensors(m))
    assert any("convs" in n for n in names) and any("gru.weight_ih_l0" in n for n in names)
    assert any("gru.weight_hh_l0_reverse" in n for n in names) and "proj.weight" in names
    assert not any("bias" in n or "bn" in n for n in names)
    for sp in (0.3, 0.5, 0.7):
        mm = build_model({"name": "crnn"})
        masks = global_magnitude_masks(mm, sp)
        apply_masks(mm, masks)
        rep = sparsity_report(mm)
        assert abs(rep["actual_sparsity_of_prunable"] - sp) < 1e-3
        assert rep["nonzero_params"] < rep["total_params"]


def test_pruning_is_global_not_per_layer():
    m = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
    with torch.no_grad():
        m[0].weight.mul_(10.0)             # layer 0 has much larger weights
    masks = global_magnitude_masks(m, 0.5)
    kept0, kept1 = masks["0.weight"].float().mean().item(), masks["1.weight"].float().mean().item()
    assert kept0 > 0.8 and kept1 < 0.3 and abs(kept0 + kept1 - 1.0) < 0.05   # per-layer pruning would keep ~50% of each


def test_masks_survive_optimizer_steps_when_reapplied():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    masks = global_magnitude_masks(m, 0.6)
    apply_masks(m, masks)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.01)
    for _ in range(5):
        loss = m(torch.randn(16, 8)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        apply_masks(m, masks)
    for n, w in prunable_tensors(m).items():
        assert (w[~masks[n]] == 0).all()
    assert abs(sparsity_report(m)["actual_sparsity_of_prunable"] - 0.6) < 0.02


def test_gzip_size_shrinks_with_sparsity_but_dense_size_does_not():
    torch.manual_seed(0)
    m = build_model({"name": "cnn_baseline"})
    before = sizes(m)
    apply_masks(m, global_magnitude_masks(m, 0.7))
    after = sizes(m)
    assert after["dense_fp32_bytes"] == before["dense_fp32_bytes"]
    assert after["gzip_bytes"] < 0.5 * before["gzip_bytes"]


def test_finetune_lr_rule_and_ast_prunable_set():
    assert finetune_lr(1e-3) == pytest.approx(1e-4) and finetune_lr(3e-4) == pytest.approx(3e-5)
    assert finetune_lr(5e-5) == pytest.approx(5e-6)
    from transformers import ASTConfig
    from src.models.ast import ASTClassifier
    import src.models.ast as A
    cfg = ASTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                    max_length=400, num_labels=10)
    from transformers import ASTForAudioClassification
    tiny = ASTForAudioClassification(cfg)
    names = prunable_tensors(tiny).keys()
    assert any("q_proj" in n for n in names) and any("patch_embeddings.projection" in n for n in names)
    assert not any("position_embeddings" in n or "layernorm" in n.lower() for n in names)
