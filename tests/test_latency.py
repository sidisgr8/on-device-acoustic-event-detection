import importlib.util
import statistics
import sys
from pathlib import Path

import numpy as np
import pytest

from src.benchmark import latency as L

PS_HEADER = "  PID COMMAND\n"


def test_summarize_statistics():
    t = np.arange(1, 201) / 1000.0           # 1..200 ms
    s = L.summarize(t)
    assert s["n"] == 200 and s["median_ms"] == pytest.approx(100.5) and s["min_ms"] == pytest.approx(1)
    assert s["p95_ms"] == pytest.approx(np.percentile(np.arange(1, 201), 95))
    assert s["max_ms"] == pytest.approx(200) and s["mean_ms"] == pytest.approx(100.5)
    assert s["std_ms"] == pytest.approx(statistics.stdev(range(1, 201)))
    assert L.summarize([0.002])["std_ms"] == 0.0


def test_measure_runs_exactly_warmup_plus_timed_and_times_only_the_timed_calls():
    calls = []
    clock = iter(range(1000))                      # fake timer: +1 per call
    out = L.measure(lambda: calls.append(1), warmup=3, runs=5, timer=lambda: next(clock))
    assert len(calls) == 8 and len(out) == 5
    assert all(o == 1 for o in out)                # start/stop are consecutive fake ticks


def test_protocol_constants_match_spec():
    assert L.WARMUP == 20 and L.RUNS == 200 and L.THREADS == (1, 4)


def test_idle_machine_passes():
    ps = PS_HEADER + "  1 /sbin/launchd\n  500 /Applications/Safari.app\n"
    assert L.check_idle(ps_output=ps, loadavg=0.3) == []


@pytest.mark.parametrize("cmd", ["python scripts/run_queue.py --queue x", "python scripts/train.py --config y",
                                 "python scripts/prune.py", "python scripts/quant_eval.py", "python -m pytest"])
def test_busy_when_job_is_running(cmd):
    reasons = L.check_idle(ps_output=PS_HEADER + f"  4242 {cmd}\n", loadavg=0.1)
    assert reasons and "4242" in reasons[0]


def test_busy_when_load_is_high_and_own_process_is_ignored():
    r = L.check_idle(ps_output=PS_HEADER, loadavg=3.5)
    assert len(r) == 1 and "load average" in r[0]
    assert L.check_idle(ps_output=PS_HEADER + "  77 python scripts/benchmark_latency.py\n", loadavg=0.1) == []
    assert L.check_idle(ps_output=PS_HEADER + "  99 python scripts/run_queue.py\n", loadavg=0.1, own_pid=99) == []


def test_metadata_records_machine_and_caveat():
    m = L.machine_metadata({"x": 1})
    for k in ("date", "cpu", "torch", "torch_default_threads", "performance_cores", "efficiency_cores",
              "protocol", "caveat", "load_average_at_start", "x"):
        assert k in m
    assert "affinity" in m["caveat"] and m["protocol"]["threads"] == [1, 4]


def test_cli_refuses_when_busy_with_exit_code_2(monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("bench_cli", Path(__file__).resolve().parents[1] / "scripts" / "benchmark_latency.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "check_idle", lambda: ["process running: run_queue.py (pid 1)"])
    monkeypatch.setattr(sys, "argv", ["benchmark_latency.py"])
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert e.value.code == 2
    assert "REFUSING" in capsys.readouterr().err


def test_repeat_orders_differ_and_cover_all_models():
    exps = ["a", "b", "c", "d", "e"]
    orders = L.repeat_orders(exps, 3)
    assert orders[0] == exps and orders[1] == exps[::-1]
    assert all(sorted(o) == sorted(exps) for o in orders)
    assert len({o[0] for o in orders}) == 3 and len({o[-1] for o in orders}) == 3     # nobody always first / last


def test_aggregate_repeats_median_of_medians_and_spread_flag():
    def rec(m, p, th=1, exp="x"):
        return {"experiment": exp, "variant": "v", "threads": th, "stage": "forward", "size_bytes": 5, "repeat": 1,
                "median_ms": m, "p95_ms": p, "mean_ms": m, "std_ms": 0.1, "min_ms": m, "max_ms": p, "n": 200}
    reps = [[rec(10.0, 12.0), rec(5.0, 6.0, exp="y")], [rec(10.5, 13.0), rec(5.0, 6.0, exp="y")],
            [rec(12.0, 14.0), rec(5.2, 6.5, exp="y")]]
    agg = {r["experiment"]: r for r in L.aggregate_repeats(reps)}
    x, y = agg["x"], agg["y"]
    assert x["median_ms"] == 10.5 and x["p95_ms"] == 13.0 and x["n_repeats"] == 3 and x["repeat_medians_ms"] == [10.0, 10.5, 12.0]
    assert x["spread_pct"] == pytest.approx(100 * 2.0 / 10.5) and x["noisy"] is True      # 19 % > 10 %
    assert y["median_ms"] == 5.0 and y["noisy"] is False and "repeat" not in x and x["size_bytes"] == 5


def test_aggregate_repeats_keeps_failures():
    bad = {"experiment": "x", "variant": "v", "threads": 1, "stage": "forward", "error": "boom"}
    out = L.aggregate_repeats([[bad], [bad]])
    assert len(out) == 1 and out[0]["n_repeats"] == 0 and out[0]["error"] == "boom"
