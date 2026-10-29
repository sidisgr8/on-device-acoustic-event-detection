"""Latency harness logic (spec 4.6): batch 1, one 4 s clip, CPU only, 20 warm-up + 200 timed runs,
median and p95, at 1 and 4 threads, feature extraction timed separately from the forward pass.

Everything here is import-safe and has no side effects; the CLI is scripts/benchmark_latency.py.
The harness REFUSES to run unless the machine is idle (`check_idle`)."""
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

WARMUP = 20
RUNS = 200
THREADS = (1, 4)
MAX_LOAD_1MIN = 2.0   # D-41, D-42: was 1.0; macOS background daemons keep the 1-minute load at 1-2 on this machine
BUSY_PATTERNS = ("run_queue.py", "run_post.py", "scripts/train.py", "scripts/prune.py",
                 "scripts/quant_eval.py", "pytest", "scripts/build_cache.py", "scripts/eda.py")

CHIP_CAVEAT = ("This Mac has a heterogeneous CPU (performance + efficiency cores) and macOS gives no CPU "
               "affinity control, so 'N threads' means N worker threads whose cores are chosen by the OS "
               "scheduler. 4 threads equals the number of performance cores but is not guaranteed to run on "
               "them; multi-thread numbers are noisier than single-thread ones. Tiny models can be slower "
               "with more threads (synchronisation overhead). An Apple CPU is not an edge-device CPU, so "
               "absolute latencies are a relative comparison between models, not deployment predictions.")


def summarize(times_s) -> dict:
    """Milliseconds: median, p95 (linear-interpolated percentile), mean, std, min, max, n."""
    t = np.asarray(times_s, dtype=np.float64) * 1000.0
    return {"median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)),
            "mean_ms": float(t.mean()), "std_ms": float(t.std(ddof=1)) if len(t) > 1 else 0.0,
            "min_ms": float(t.min()), "max_ms": float(t.max()), "n": int(len(t))}


def measure(fn, warmup=WARMUP, runs=RUNS, timer=time.perf_counter) -> list:
    """Call fn() `warmup` times untimed, then `runs` times timed; return per-call seconds."""
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(runs):
        t0 = timer()
        fn()
        out.append(timer() - t0)
    return out


def _ancestors(pid) -> set:
    out, seen = set(), pid
    while seen > 1:
        r = subprocess.run(["ps", "-o", "ppid=", "-p", str(seen)], capture_output=True, text=True).stdout.strip()
        seen = int(r) if r.isdigit() else 0
        if seen > 1:
            out.add(seen)
    return out


def check_idle(ps_output: str | None = None, loadavg: float | None = None,
               max_load=MAX_LOAD_1MIN, own_pid: int | None = None) -> list:
    """Return the list of reasons the machine is NOT idle (empty list = idle).

    ps_output/loadavg are injectable for tests."""
    reasons = []
    live = ps_output is None
    if ps_output is None:
        ps_output = subprocess.run(["ps", "-axo", "pid,command"], capture_output=True, text=True).stdout
    own_pid = os.getpid() if own_pid is None else own_pid
    skip = {own_pid} | (_ancestors(own_pid) if live else set())   # e.g. scripts/run_post.py --stage latency is our parent
    for line in ps_output.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) < 2 or not parts[0].isdigit() or int(parts[0]) in skip:
            continue
        cmd = parts[1]
        if "benchmark_latency.py" in cmd:
            continue
        for pat in BUSY_PATTERNS:
            if pat in cmd and "grep" not in cmd:
                reasons.append(f"process running: {pat} (pid {parts[0]})")
                break
    if loadavg is None:
        loadavg = os.getloadavg()[0]
    if loadavg > max_load:
        reasons.append(f"1-minute load average {loadavg:.2f} > {max_load} (wait a few minutes and retry)")
    return reasons


def _sysctl(name):
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def machine_metadata(extra: dict | None = None) -> dict:
    import torch
    meta = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "cpu": _sysctl("machdep.cpu.brand_string"), "platform": platform.platform(),
        "logical_cpus": os.cpu_count(), "performance_cores": _sysctl("hw.perflevel0.logicalcpu"),
        "efficiency_cores": _sysctl("hw.perflevel1.logicalcpu"), "memory_bytes": _sysctl("hw.memsize"),
        "python": sys.version.split()[0], "torch": torch.__version__,
        "torch_default_threads": torch.get_num_threads(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "load_average_at_start": os.getloadavg(),
        "protocol": {"batch_size": 1, "clip_seconds": 4.0, "warmup_runs": WARMUP, "timed_runs": RUNS,
                     "threads": list(THREADS), "device": "cpu",
                     "percentile_method": "numpy linear interpolation"},
        "caveat": CHIP_CAVEAT,
    }
    try:
        import onnxruntime
        meta["onnxruntime"] = onnxruntime.__version__
    except Exception:
        pass
    if extra:
        meta.update(extra)
    return meta


SPREAD_FLAG_PCT = 10.0


def repeat_orders(experiments, repeats=3) -> list:
    """Model order per repeat: given order, reversed, then rotations by r. Spreads any heating/ordering effect."""
    exps = list(experiments)
    orders = []
    for r in range(repeats):
        if r == 0:
            orders.append(exps)
        elif r == 1:
            orders.append(exps[::-1])
        else:
            k = r % len(exps)
            orders.append(exps[k:] + exps[:k])
    return orders


def aggregate_repeats(repeat_records: list, key=("experiment", "variant", "threads", "stage")) -> list:
    """Combine the records of several full repeats (list of lists of dicts) into one record per measurement.

    median_ms / p95_ms of the result are the MEDIAN across repeats of each repeat's median / p95; the spread is
    (max - min) / median over the repeat medians, in percent; spread above SPREAD_FLAG_PCT sets noisy=True."""
    groups = {}
    for recs in repeat_records:
        for r in recs:
            groups.setdefault(tuple(r.get(k) for k in key), []).append(r)
    out = []
    for g in groups.values():
        ok = [r for r in g if "median_ms" in r]
        if not ok:
            out.append({**g[0], "n_repeats": 0})
            continue
        med = [r["median_ms"] for r in ok]
        m = float(np.median(med))
        spread = 100.0 * (max(med) - min(med)) / m if m > 0 else 0.0
        agg = {k: v for k, v in ok[0].items() if k not in ("median_ms", "p95_ms", "mean_ms", "std_ms", "min_ms", "max_ms", "n", "repeat")}
        agg.update(median_ms=m, p95_ms=float(np.median([r["p95_ms"] for r in ok])), repeat_medians_ms=med,
                   repeat_p95_ms=[r["p95_ms"] for r in ok], spread_pct=spread, noisy=spread > SPREAD_FLAG_PCT,
                   n_repeats=len(ok), n_per_repeat=ok[0]["n"])
        out.append(agg)
    return out


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"unavailable: {e}"


def machine_state() -> dict:
    """Load average, power source and thermal state right now (macOS pmset)."""
    return {"time": datetime.now().isoformat(timespec="seconds"), "load_average": os.getloadavg(),
            "pmset_batt": _run(["pmset", "-g", "batt"]), "pmset_therm": _run(["pmset", "-g", "therm"])}
