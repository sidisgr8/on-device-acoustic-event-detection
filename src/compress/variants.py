"""Compressed / deployment variants of a trained model, behind one `Runner` interface.

Variants (D-09: report PyTorch int8 AND ONNX Runtime int8, always with size and accuracy separately):

  fp32_torch              eager PyTorch float32 (reference, CPU)
  torch_dynamic_int8      torch.ao dynamic int8 for Linear (+GRU for the CRNN): CRNN, AST
  torch_static_int8_fx    torch.ao FX-graph-mode static int8 (qnnpack) with 100 calibration clips: CNN, MobileNetV3
  onnx_fp32               ONNX export, ONNX Runtime float32 (checked numerically against PyTorch first)
  onnx_int8               ORT static QDQ int8 (conv models, 100 calibration clips) or ORT dynamic int8 (CRNN, AST)

A variant that fails to build is recorded as a finding (error text), never retried silently.
"""
import copy
import io
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

ENGINE = "qnnpack"  # the only quantized backend in this torch build (arm64)
ONNX_OPSET = 17
ONNX_TOL = 1e-3     # max |logit difference| ONNX fp32 vs PyTorch fp32 required before quantizing

STATIC_KINDS = ("cnn_baseline", "mobilenetv3_small")


def variants_for(kind: str):
    torch_q = "torch_static_int8_fx" if kind in STATIC_KINDS else "torch_dynamic_int8"
    return ["fp32_torch", torch_q, "onnx_fp32", "onnx_int8"]


class Runner:
    """Callable: float tensor batch -> logits tensor. Also knows its on-disk size."""

    def __init__(self, name, backend, fn, size_bytes, meta=None):
        self.name, self.backend, self.fn, self.size_bytes = name, backend, fn, size_bytes
        self.meta = meta or {}

    def __call__(self, x):
        return self.fn(x)


def state_bytes(module: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return buf.tell()


def fp32_runner(model) -> Runner:
    model = model.eval()
    return Runner("fp32_torch", "pytorch", lambda x: model(x), state_bytes(model))


def dynamic_int8(model, kind: str, path=None) -> Runner:
    torch.backends.quantized.engine = ENGINE
    layers = {nn.Linear, nn.GRU} if kind == "crnn" else {nn.Linear}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = torch.ao.quantization.quantize_dynamic(copy.deepcopy(model).eval(), layers, dtype=torch.qint8)
    if path:
        torch.save(q, path)
    return Runner("torch_dynamic_int8", "pytorch", lambda x: q(x), state_bytes(q),
                  {"quantized_layers": sorted(l.__name__ for l in layers), "engine": ENGINE})


def static_int8_fx(model, calib_x: torch.Tensor, path=None) -> Runner:
    torch.backends.quantized.engine = ENGINE
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qmap = get_default_qconfig_mapping(ENGINE)
        prepared = prepare_fx(copy.deepcopy(model).eval(), qmap, (calib_x[:1],))
        with torch.no_grad():
            for s in range(0, len(calib_x), 10):
                prepared(calib_x[s:s + 10])
        q = convert_fx(prepared)
        if path:
            # A pickled FX-quantized module cannot be reloaded in this torch (fused ConvReLU2d fails to
            # unpickle), so the artifact is TorchScript, which is also what a mobile deployment would ship.
            torch.jit.save(torch.jit.trace(q, calib_x[:1]), str(path))
    return Runner("torch_static_int8_fx", "pytorch", lambda x: q(x), state_bytes(q),
                  {"engine": ENGINE, "n_calibration": len(calib_x)})


# ---------------------------------------------------------------- ONNX

def export_onnx(model, example: torch.Tensor, path: Path):
    model = model.eval()
    kw = dict(input_names=["input"], output_names=["logits"], opset_version=ONNX_OPSET,
              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.onnx.export(model, (example,), str(path), dynamo=False, **kw)  # TorchScript exporter
            return "torchscript"
        except Exception as first:
            torch.onnx.export(model, (example,), str(path), dynamo=True, **{
                k: v for k, v in kw.items() if k != "dynamic_axes"},
                dynamic_shapes={"x": {0: torch.export.Dim("batch")}})
            return f"dynamo (torchscript failed: {first!r:.80})"


def ort_session(path, threads=1):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def ort_runner(name, path, threads=1, meta=None) -> Runner:
    sess = ort_session(path, threads)
    return Runner(name, "onnxruntime", lambda x: torch.from_numpy(
        sess.run(None, {"input": x.numpy()})[0]), os.path.getsize(path), meta)


def onnx_check(model, onnx_path, check_x: torch.Tensor) -> dict:
    """ONNX fp32 must match PyTorch fp32 before anything is quantized."""
    sess = ort_session(onnx_path)
    with torch.no_grad():
        ref = model.eval()(check_x).numpy()
    out = sess.run(None, {"input": check_x.numpy()})[0]
    diff = float(np.abs(ref - out).max())
    return {"max_abs_logit_diff": diff, "argmax_agreement": float((ref.argmax(1) == out.argmax(1)).mean()),
            "n": int(len(check_x)), "tol": ONNX_TOL, "ok": diff <= ONNX_TOL}


class _CalibReader:
    def __init__(self, x: torch.Tensor):
        self.it = iter([{"input": x[i:i + 1].numpy()} for i in range(len(x))])

    def get_next(self):
        return next(self.it, None)


def ort_int8(onnx_fp32: Path, out_path: Path, kind: str, calib_x: torch.Tensor) -> str:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic, quantize_static
    if kind in STATIC_KINDS:
        src = onnx_fp32
        pre = onnx_fp32.with_name(onnx_fp32.stem + "_pre.onnx")
        try:
            from onnxruntime.quantization.shape_inference import quant_pre_process
            quant_pre_process(str(onnx_fp32), str(pre))
            src = pre
        except Exception:
            pass
        from onnxruntime.quantization import CalibrationDataReader

        class Reader(_CalibReader, CalibrationDataReader):
            pass
        quantize_static(str(src), str(out_path), Reader(calib_x), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
        return "ORT static QDQ int8 (activations uint8, weights int8), 100 calibration clips"
    quantize_dynamic(str(onnx_fp32), str(out_path), weight_type=QuantType.QInt8)
    note = "ORT dynamic int8 (weights int8, MatMul/Gemm)"
    if kind == "crnn":
        note += "; ORT has no dynamic int8 for GRU, so the GRU stays float32"
    return note


def build_variants(kind, model, calib_x, check_x, workdir: Path, log=print):
    """Build every variant for one trained model. Returns {name: {'runner'|'error', ...}}.
    Artifacts (pickled quantized modules, .onnx files) are written under workdir for the latency tool."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out = {}

    def attempt(name, fn):
        try:
            out[name] = {"runner": fn()}
            log(f"    {name}: ok")
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"[:600]}
            log(f"    {name}: FAILED {out[name]['error'][:160]}")

    attempt("fp32_torch", lambda: fp32_runner(model))
    if kind in STATIC_KINDS:
        attempt("torch_static_int8_fx", lambda: static_int8_fx(model, calib_x, workdir / "torch_static_int8_fx.pt"))
    else:
        attempt("torch_dynamic_int8", lambda: dynamic_int8(model, kind, workdir / "torch_dynamic_int8.pt"))

    onnx32 = workdir / "model_fp32.onnx"
    try:
        exporter = export_onnx(model, check_x[:1], onnx32)
        chk = onnx_check(model, onnx32, check_x)
        log(f"    onnx export ({exporter}); max |logit diff| vs PyTorch = {chk['max_abs_logit_diff']:.2e}")
        if not chk["ok"]:
            raise RuntimeError(f"ONNX fp32 does not match PyTorch: {chk}")
        r = ort_runner("onnx_fp32", onnx32, meta={"exporter": exporter, "check": chk})
        out["onnx_fp32"] = {"runner": r}
        attempt("onnx_int8", lambda: _onnx_int8(kind, onnx32, workdir, calib_x))
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"[:600]
        out["onnx_fp32"] = {"error": msg}
        out["onnx_int8"] = {"error": "skipped: ONNX fp32 export/check failed"}
        log(f"    onnx_fp32: FAILED {msg[:160]}")
    return out


def _onnx_int8(kind, onnx32, workdir, calib_x):
    path = workdir / "model_int8.onnx"
    note = ort_int8(onnx32, path, kind, calib_x)
    return ort_runner("onnx_int8", path, meta={"quantization": note})


def load_runner(variant: str, kind: str, workdir: Path, model_fp32=None, threads=1) -> Runner:
    """Rebuild a runner from artifacts saved by `build_variants` (used by the latency harness)."""
    workdir = Path(workdir)
    if variant == "fp32_torch":
        return fp32_runner(model_fp32)
    if variant.startswith("torch_"):
        torch.backends.quantized.engine = ENGINE
        path = workdir / f"{variant}.pt"
        if variant == "torch_static_int8_fx":
            q = torch.jit.load(str(path)).eval()
            return Runner(variant, "pytorch", lambda x: q(x), os.path.getsize(path), {"artifact": "torchscript"})
        q = torch.load(path, weights_only=False).eval()
        return Runner(variant, "pytorch", lambda x: q(x), state_bytes(q))
    path = workdir / ("model_fp32.onnx" if variant == "onnx_fp32" else "model_int8.onnx")
    return ort_runner(variant, path, threads=threads)
