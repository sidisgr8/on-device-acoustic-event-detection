"""Show queue progress: done / running / pending / failed, mean accuracy so far, ETA.

    python scripts/status.py [--queue configs/queue_main.yaml]
"""
import argparse
import json
import shutil
import statistics as st
import time
from datetime import datetime, timedelta
from pathlib import Path

from src.train.queue import load_queue, pid_alive, read_state


def fmt(sec):
    sec = int(max(sec, 0))
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="configs/queue_main.yaml")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--logs-dir", default="logs")
    a = ap.parse_args()
    jobs = load_queue(a.queue)
    state = read_state(a.logs_dir)
    alive = pid_alive(state.get("pid"))
    cur = state.get("current") if alive else None
    failures = []
    fp = Path(a.logs_dir) / "failures.jsonl"
    if fp.exists():
        failures = [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]

    head = {True: "RUNNING", False: "NOT RUNNING"}[alive]
    if state:
        head += f"  (pid {state.get('pid')}, started {state.get('started')}, state={state.get('status')}, " \
                f"commit {str((state.get('git') or {}).get('commit'))[:8]})"
    print(f"Queue: {head}\n")
    total_eta, eta_note = 0.0, False
    print(f"{'experiment':30s} {'done':>5s} {'acc mean±std':>14s} {'macroF1':>8s} {'s/fold':>7s}  pending / running / failed")
    probe = Path(a.results_dir) / "ast_amp_probe.json"
    for j in jobs:
        if j.get("stage"):
            if probe.exists():
                print(f"{'[stage] ' + j['stage']:30s} done -> AST precision {json.loads(probe.read_text())['chosen']}")
            else:
                running_probe = cur is not None and cur["experiment"] == "amp_probe"
                print(f"{'[stage] ' + j['stage']:30s} {'RUNNING' if running_probe else 'pending'}")
                total_eta += 240
            continue
        exp, folds = j["experiment"], j["folds"]
        res = {}
        for k in folds:
            p = Path(a.results_dir) / exp / f"fold{k}.json"
            if p.exists():
                r = json.loads(p.read_text())
                if r.get("protocol_version") == 2:  # protocol v1 results are never shown
                    res[k] = r
        accs = [r["accuracy"] for r in res.values()]
        f1s = [r["macro_f1"] for r in res.values()]
        wall = [r["train_time_s"] for r in res.values()]
        acc_s = f"{100 * st.mean(accs):.2f}±{100 * (st.stdev(accs) if len(accs) > 1 else 0):.2f}" if accs else "-"
        f1_s = f"{st.mean(f1s):.3f}" if f1s else "-"
        max_ep = j["cfg"]["train"]["max_epochs"]
        if wall:
            per_fold, upper = st.mean(wall) * 1.03, False
        else:
            per_fold, upper = (j["est_epoch_s"] or 0) * max_ep, True
        running = cur["fold"] if cur and cur["experiment"] == exp else None
        failed = sorted({f["fold"] for f in failures if f["experiment"] == exp and f["fold"] not in res})
        pending = [k for k in folds if k not in res and k != running]
        run_txt = ""
        eta_job = per_fold * len(pending)
        if running is not None:
            ep_file = Path(a.results_dir) / exp / f"fold{running}_epochs.jsonl"
            eps = [json.loads(l) for l in ep_file.read_text().splitlines()] if ep_file.exists() else []
            elapsed = sum(e["epoch_total_s"] for e in eps)
            run_txt = f"fold {running} (epoch {len(eps)}/{max_ep}" + (f", val acc {eps[-1]['val_acc']:.3f}" if eps else "") + ")"
            eta_job += max(per_fold - elapsed, 0)
        total_eta += eta_job
        eta_note |= upper and bool(pending or running)
        print(f"{exp:30s} {len(res):>2d}/{len(folds):<2d} {acc_s:>14s} {f1_s:>8s} {per_fold:>7.0f}"
              f"  pend {pending if len(pending) < 11 else len(pending)}"
              + (f" | running {run_txt}" if run_txt else "") + (f" | FAILED {failed}" if failed else ""))
    print()
    if total_eta > 0:
        done_at = datetime.now() + timedelta(seconds=total_eta)
        print(f"ETA to finish queue: ~{fmt(total_eta)} (around {done_at:%a %d %b %H:%M})"
              + ("  [estimate: experiments with no finished fold use measured s/epoch x fixed epochs]" if eta_note else ""))
    else:
        print("Nothing pending.")
    ck = Path("checkpoints")
    best_bytes = sum(f.stat().st_size for f in ck.glob("*/fold*/best.pt")) if ck.exists() else 0
    free = shutil.disk_usage(".").free
    print(f"\nDisk: best.pt checkpoints {best_bytes / 1e9:.2f} GB (kept), {free / 1e9:.0f} GB free")
    if failures:
        last = failures[-1]
        print(f"\n{len(failures)} failure record(s); latest: {last['experiment']} fold {last['fold']} {last['error']} (see logs/failures.jsonl)")
    if not alive and total_eta > 0:
        print("\nQueue is not running. Resume with:  scripts/launch_queue.sh")


if __name__ == "__main__":
    main()
