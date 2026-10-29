import numpy as np
import pytest
import torch

from src.models import build_model

X = torch.randn(2, 1, 128, 201)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def test_crnn_shape_claims_from_the_spec():
    m = build_model({"name": "crnn"})
    out, s = m(X, return_shapes=True)
    assert s["after_conv"] == (2, 128, 16, 50)          # frequency 128 -> 16, 50 time steps
    assert s["flattened_per_step"] == (2, 50, 2048)     # 128 channels x 16 bins
    assert s["gru_out"] == (2, 50, 256) and s["pooled"] == (2, 512)
    assert out.shape == (2, 10)
    print("CRNN params", n_params(m))
    assert 500_000 < n_params(m) < 650_000


def test_mobilenet_shapes_and_params():
    m = build_model({"name": "mobilenetv3_small", "pretrained": False})
    assert m(X).shape == (2, 10)
    assert m.net.conv_stem.weight.shape[1] == 1          # single-channel stem
    assert m.net.get_classifier().out_features == 10
    print("MobileNetV3-Small params", n_params(m))
    assert 1_400_000 < n_params(m) < 1_700_000


def test_cnn_params():
    print("CNN params", n_params(build_model({"name": "cnn_baseline"})))


def test_one_channel_stem_is_the_sum_of_rgb_filters():
    """With pretrained RGB weights, the 1-channel stem must equal the sum of the 3 filters,
    so a grey image fed to the 3-channel model gives the same stem output."""
    timm = pytest.importorskip("timm")
    try:
        m3 = timm.create_model("mobilenetv3_small_100", pretrained=True, in_chans=3)
        m1 = build_model({"name": "mobilenetv3_small", "pretrained": True})
    except Exception as e:  # offline
        pytest.skip(f"pretrained weights unavailable: {e}")
    w3, w1 = m3.conv_stem.weight, m1.net.conv_stem.weight
    torch.testing.assert_close(w1, w3.sum(dim=1, keepdim=True))
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(m1.net.conv_stem(x), m3.conv_stem(x.repeat(1, 3, 1, 1)),
                                   rtol=1e-4, atol=1e-4)
    prov = m1.provenance
    assert prov["hf_hub_id"] and prov.get("sha256")


def test_models_are_finite_and_trainable_one_step():
    for cfg in ({"name": "cnn_baseline"}, {"name": "crnn"}, {"name": "mobilenetv3_small"}):
        m = build_model(cfg)
        loss = torch.nn.functional.cross_entropy(m(X), torch.tensor([1, 2]))
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
