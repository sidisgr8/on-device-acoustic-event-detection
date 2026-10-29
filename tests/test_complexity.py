import pytest
import torch
from torch import nn

from src.analysis import complexity as C


def test_hook_counts_match_hand_calculation():
    conv = nn.Conv2d(3, 8, 3, padding=1)               # out 8x10x10 -> 800 outputs x (3*3*3) MACs
    assert C.count(conv, torch.randn(1, 3, 10, 10))["conv"] == 8 * 10 * 10 * 27
    lin = nn.Linear(16, 4)
    assert C.count(lin, torch.randn(1, 16))["linear"] == 64
    assert C.count(lin, torch.randn(1, 5, 16))["linear"] == 5 * 64          # applied per token
    gru = nn.GRU(8, 6, batch_first=True, bidirectional=True)
    assert C.count(gru, torch.randn(1, 7, 8))["gru"] == 7 * 2 * 3 * (8 * 6 + 6 * 6)
    dw = nn.Conv2d(4, 4, 3, padding=1, groups=4)                            # depthwise: one input channel per filter
    assert C.count(dw, torch.randn(1, 4, 6, 6))["conv"] == 4 * 6 * 6 * 9


def test_attention_macs_formula():
    assert C.attention_macs(12, 470, 768) == 12 * 2 * 470 * 470 * 768
    assert C.AST_TOKENS == 470


@pytest.mark.parametrize("exp,params,macs_m", [("cnn_baseline", 390_890, 361.3), ("crnn", 558_442, 266.3),
                                              ("mobilenetv3_small_scratch", 1_527_818, 28.6)])
def test_real_models_match_the_documented_numbers(exp, params, macs_m):
    torch.set_num_threads(1)
    r = C.model_complexity(exp)
    assert r["params"] == params
    assert r["macs_total"] / 1e6 == pytest.approx(macs_m, abs=0.1)
    assert r["macs_attention"] == 0


def test_cnn_block_macs_are_balanced_and_pretraining_does_not_change_macs():
    """Channels double while resolution halves, so blocks 2-4 cost the same: the CNN is parameter-light but not MAC-light."""
    from src.models import build_model
    m = build_model({"name": "cnn_baseline"})
    per_block = []
    x = torch.randn(1, 1, 128, 201)
    for blk in m.features:
        per_block.append(C.count(blk[0], x)["conv"])
        x = blk(x)
    assert per_block[0] < 0.1 * per_block[1]
    assert per_block[1] == per_block[2] == per_block[3]
