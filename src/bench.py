"""Efficiency harness: parameters, model size, CPU latency (Report Section 4.6).

    python -m src.bench --model cnn
Writes results/<model>/bench.json. Run ALL models on the SAME machine so the
latency numbers are comparable.
"""
import argparse
import io
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from .models import build_model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)               # fp32 weights as saved to disk
    return buf.tell() / 1e6


@torch.no_grad()
def latency_ms(model, threads, warmup, runs):
    torch.set_num_threads(threads)
    model.eval()
    x = torch.randn(1, 1, 128, 201)                    # one 4 s clip, batch size 1
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter(); model(x); times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return {"median_ms": statistics.median(times), "p95_ms": times[int(0.95 * len(times)) - 1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    model = build_model(args.model).cpu()               # CPU only, always
    res = {
        "model": args.model, "params_millions": count_params(model) / 1e6,
        "size_mb_fp32": size_mb(model),
        "latency_1_thread": latency_ms(model, 1, args.warmup, args.runs),
        "latency_4_threads": latency_ms(model, 4, args.warmup, args.runs),
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "torch": torch.__version__, "cpu_count": __import__("os").cpu_count()},
        "note": "feature extraction (log-mel) is NOT included in these timings",
    }
    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
