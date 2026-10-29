import numpy as np
import pytest
import torch
from torch import nn

ort = pytest.importorskip("onnxruntime")   # only installed in .venv-post

from src.compress import variants as V
from src.compress.common import evaluate_runner
from src.models import build_model


class TinyConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4), nn.ReLU(),
                               nn.MaxPool2d(2), nn.Conv2d(4, 8, 3, padding=1), nn.ReLU())
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        return self.fc(self.c(x).mean(dim=(2, 3)))


def calib(n=12, shape=(1, 16, 20)):
    torch.manual_seed(0)
    return torch.randn(n, *shape)


def test_static_fx_and_onnx_on_a_small_conv_net(tmp_path):
    torch.manual_seed(0)
    m = TinyConv().eval()
    x = calib()
    b = V.build_variants("cnn_baseline", m, x, x[:8], tmp_path)
    assert set(b) == {"fp32_torch", "torch_static_int8_fx", "onnx_fp32", "onnx_int8"}
    assert all("runner" in v for v in b.values()), {k: v.get("error") for k, v in b.items()}
    ref = m(x[:8])
    for name, tol in (("onnx_fp32", 1e-4), ("torch_static_int8_fx", 0.25), ("onnx_int8", 0.25)):
        out = b[name]["runner"](x[:8])
        assert out.shape == (8, 10) and float((out - ref).abs().max()) < tol, name
    assert b["torch_static_int8_fx"]["runner"].size_bytes < b["fp32_torch"]["runner"].size_bytes
    assert b["onnx_fp32"]["runner"].meta["check"]["ok"]       # (ORT QDQ adds fixed overhead: no size claim for a toy net)
    assert (tmp_path / "model_int8.onnx").exists() and (tmp_path / "torch_static_int8_fx.pt").exists()


def test_crnn_dynamic_and_onnx(tmp_path):
    torch.manual_seed(0)
    m = build_model({"name": "crnn"}).eval()
    x = torch.randn(6, 1, 128, 201)
    b = V.build_variants("crnn", m, x, x[:4], tmp_path)
    assert all("runner" in v for v in b.values()), {k: v.get("error") for k, v in b.items()}
    ref = m(x[:4])
    assert float((b["torch_dynamic_int8"]["runner"](x[:4]) - ref).abs().max()) < 0.3
    assert "GRU stays float32" in b["onnx_int8"]["runner"].meta["quantization"]
    assert b["torch_dynamic_int8"]["runner"].size_bytes < 0.6 * b["fp32_torch"]["runner"].size_bytes


def test_onnx_check_gates_quantization(tmp_path, monkeypatch):
    m = TinyConv().eval()
    x = calib()
    monkeypatch.setattr(V, "ONNX_TOL", 0.0)                  # impossible tolerance -> must refuse to quantize
    monkeypatch.setattr(V, "onnx_check", lambda *a, **k: {"ok": False, "max_abs_logit_diff": 1.0})
    b = V.build_variants("cnn_baseline", m, x, x[:4], tmp_path)
    assert "error" in b["onnx_fp32"] and "skipped" in b["onnx_int8"]["error"]


def test_a_failing_variant_is_recorded_as_a_finding_not_raised(tmp_path, monkeypatch):
    m = TinyConv().eval()
    x = calib()
    monkeypatch.setattr(V, "static_int8_fx", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no fx")))
    b = V.build_variants("cnn_baseline", m, x, x[:4], tmp_path)
    assert "no fx" in b["torch_static_int8_fx"]["error"] and "runner" in b["onnx_int8"]


def test_load_runner_roundtrip_from_saved_artifacts(tmp_path):
    m = TinyConv().eval()
    x = calib()
    built = V.build_variants("cnn_baseline", m, x, x[:4], tmp_path)
    for v in ("torch_static_int8_fx", "onnx_fp32", "onnx_int8"):
        r = V.load_runner(v, "cnn_baseline", tmp_path, model_fp32=m)
        assert r(x[:2]).shape == (2, 10)
        torch.testing.assert_close(r(x[:2]), built[v]["runner"](x[:2]), rtol=1e-4, atol=1e-4)
    assert V.variants_for("ast")[1] == "torch_dynamic_int8" and V.variants_for("cnn_baseline")[1] == "torch_static_int8_fx"


def test_dynamic_int8_artifacts_roundtrip_for_crnn(tmp_path):
    torch.manual_seed(0)
    m = build_model({"name": "crnn"}).eval()
    x = torch.randn(3, 1, 128, 201)
    built = V.build_variants("crnn", m, x, x[:2], tmp_path)
    r = V.load_runner("torch_dynamic_int8", "crnn", tmp_path, model_fp32=m)
    torch.testing.assert_close(r(x[:2]), built["torch_dynamic_int8"]["runner"](x[:2]), rtol=1e-4, atol=1e-4)


def test_mobilenet_static_fx_roundtrip(tmp_path):
    torch.manual_seed(0)
    m = build_model({"name": "mobilenetv3_small", "pretrained": False}).eval()
    x = torch.randn(10, 1, 128, 201)
    built = V.build_variants("mobilenetv3_small", m, x, x[:2], tmp_path)
    assert "runner" in built["torch_static_int8_fx"], built["torch_static_int8_fx"]
    r = V.load_runner("torch_static_int8_fx", "mobilenetv3_small", tmp_path)
    torch.testing.assert_close(r(x[:2]), built["torch_static_int8_fx"]["runner"](x[:2]), rtol=1e-3, atol=1e-3)
    assert built["torch_static_int8_fx"]["runner"].size_bytes < 0.5 * built["fp32_torch"]["runner"].size_bytes


def test_evaluate_runner_metrics():
    from src.data.dataset import BatchDataset
    y = torch.arange(10) % 10
    ds = BatchDataset(torch.randn(10, 3), y, torch.arange(10).numpy())
    r = evaluate_runner(lambda x: torch.nn.functional.one_hot(y[: len(x)], 10).float(), ds, batch_size=10)
    assert r["accuracy"] == 1.0 and r["n_eval"] == 10 and r["macro_f1"] == 1.0
    wrong = evaluate_runner(lambda x: torch.nn.functional.one_hot((y[: len(x)] + 1) % 10, 10).float(), ds, batch_size=10)
    assert wrong["accuracy"] == 0.0
