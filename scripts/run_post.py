"""Run the post-training stages of configs/queue_post.yaml sequentially.

    python scripts/run_post.py                  # all non-manual stages in order
    python scripts/run_post.py --list
    python scripts/run_post.py --stage latency  # a manual stage (or re-run one stage); `--stage tables` after latency

* refuses to start while the training queue (or another stage) is running
* one log per stage in logs/post/<stage>.log; a failed stage is recorded in logs/post/failures.jsonl and the
  next stage still runs; every stage is idempotent (finished work is skipped), so re-running is safe
* after each stage, finished results are committed with scripts/commit_results.py (use --no-commit to skip)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from src.benchmark.latency import check_idle


def family():
    """This process, its ancestors and its direct children (macOS caffeinate shows up as a child of the process it wraps)."""
    rows = subprocess.run(["ps", "-axo", "pid=,ppid="], capture_output=True, text=True).stdout.split()
    ppid = {int(a): int(b) for a, b in zip(rows[::2], rows[1::2])}
    me = os.getpid()
    pids, pid = {me}, me
    while ppid.get(pid, 0) > 1 and ppid[pid] not in pids:
        pid = ppid[pid]
        pids.add(pid)
    return pids | {p for p, pp in ppid.items() if pp == me}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="configs/queue_post.yaml")
    ap.add_argument("--stage", action="append", help="run only this stage (repeatable); required for manual stages")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--logs-dir", default="logs/post")
    ap.add_argument("--no-commit", action="store_true")
    a = ap.parse_args()
    q = yaml.safe_load(open(a.queue))
    stages = q["stages"]
    if a.list:
        for s in stages:
            print(f"{s['name']:12s} resource={s.get('resource', '-'):13s} {'MANUAL' if s.get('manual') else 'auto':6s} {' '.join(map(str, s['cmd']))}")
        return
    # processes only; load is not a blocker here. Our own wrapper processes (nohup/caffeinate) are not "another job".
    mine = {str(p) for p in family()}
    busy = [r for r in check_idle(max_load=1e9) if "process running" in r and r.split("(pid ")[-1].rstrip(")") not in mine]
    if busy:
        sys.exit("REFUSING to start: " + "; ".join(busy) + "\nWait for the training queue / earlier stage to finish.")
    selected = [s for s in stages if (a.stage and s["name"] in a.stage) or (not a.stage and not s.get("manual"))]
    unknown = set(a.stage or []) - {s["name"] for s in stages}
    if unknown:
        sys.exit(f"unknown stage(s): {sorted(unknown)}")
    logs = Path(a.logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    failures = []
    for s in selected:
        cmd = [q["python"], *map(str, s["cmd"])]
        print(f"[{datetime.now():%H:%M:%S}] START {s['name']}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        with open(logs / f"{s['name']}.log", "a", buffering=1) as lf:
            lf.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} {' '.join(cmd)}\n")
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        if rc != 0:
            failures.append(s["name"])
            with open(logs / "failures.jsonl", "a") as f:
                f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "stage": s["name"],
                                    "returncode": rc, "log": str(logs / f"{s['name']}.log")}) + "\n")
            print(f"[{datetime.now():%H:%M:%S}] FAIL  {s['name']} (rc={rc}, {dt:.0f}s) see {logs / (s['name'] + '.log')}; continuing", flush=True)
        else:
            print(f"[{datetime.now():%H:%M:%S}] DONE  {s['name']} in {dt:.0f}s", flush=True)
        if not a.no_commit:
            subprocess.run([sys.executable, "scripts/commit_results.py"], check=False)
    skipped = [s["name"] for s in stages if s not in selected]
    print(f"finished. failed stages: {failures or 'none'}; not run: {skipped or 'none'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
