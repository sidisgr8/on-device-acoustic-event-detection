"""Run experiment configs and folds sequentially, unattended.

    caffeinate -dimsu nohup python scripts/run_queue.py --queue configs/queue_main.yaml \
        > logs/queue.out 2>&1 &

* skips folds that already have a result file (test folds are evaluated once)
* resumes an interrupted fold from its last.pt
* one log file per job under logs/<experiment>/fold<k>.log
* a failing job is recorded in logs/failures.jsonl and the queue moves on; re-running the
  queue retries failed folds (and resumes them)
* refuses to start from a dirty git tree: results record the commit loaded at process start
"""
import argparse
import copy
import gc
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from src.data.dataset import FeatureStore
from src.train.amp_probe import run_probe
from src.train.loop import GIT_AT_START, config_hash, pick_device, run_round
from src.train.queue import load_queue

STOP = {"flag": False}


def now():
    return datetime.now().isoformat(timespec="seconds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="configs/queue_main.yaml")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--auto-commit", action="store_true",
                    help="commit finished fold results (scripts/commit_results.py) after every "
                         "5 completed folds and at the end of each experiment")
    a = ap.parse_args()

    if GIT_AT_START["dirty"] and not a.allow_dirty:
        sys.exit("git tree is dirty: commit first so results are traceable (or --allow-dirty)")
    logs = Path(a.logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    jobs = load_queue(a.queue)
    state_path = logs / "queue_state.json"
    state = {"pid": os.getpid(), "started": now(), "queue": a.queue, "git": GIT_AT_START,
             "status": "running", "current": None, "n_done_this_run": 0, "n_failed_this_run": 0}

    def save_state():
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        os.replace(tmp, state_path)

    def on_signal(signum, _frame):
        STOP["flag"] = True
        print(f"[{now()}] signal {signum}: finishing the current epoch is NOT guaranteed; "
              "stopping now, re-run to resume", flush=True)
        state["status"] = "stopped"
        state["current"] = None
        save_state()
        os._exit(130)  # last.pt is written atomically every epoch, so resume loses <= 1 epoch

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    qlog = open(logs / "queue.log", "a", buffering=1)

    def qprint(msg):
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        qlog.write(line + "\n")

    qprint(f"queue start pid={os.getpid()} commit={GIT_AT_START['commit']} queue={a.queue}")
    save_state()
    store = None
    since_commit = 0

    def commit_results():
        if a.auto_commit:
            try:
                subprocess.run([sys.executable, "scripts/commit_results.py"], check=False, timeout=120,
                               stdout=qlog, stderr=qlog)
            except Exception as e:
                qprint(f"auto-commit skipped: {e!r}")

    probe_path = Path(a.results_dir) / "ast_amp_probe.json"
    for job in jobs:
        if job.get("stage") == "amp_probe":
            if store is None:
                store = FeatureStore(job["cfg"]["data"]["cache"])
            if probe_path.exists():
                qprint("amp_probe: result exists, skipping")
                continue
            qprint("START amp_probe (AST precision, ~30 steps per mode, fold-1 training clips only)")
            state["current"] = {"experiment": "amp_probe", "fold": 0, "started": now()}
            save_state()
            r = run_probe(job["cfg"], store, pick_device(a.device), probe_path,
                          decisions_path="docs/DECISIONS.md", log=qprint)
            qprint(f"DONE  amp_probe -> {r['chosen']} ({r['reason']})")
            state["current"] = None
            save_state()
            commit_results()
            continue
        exp, cfg = job["experiment"], copy.deepcopy(job["cfg"])
        if cfg["train"].get("amp") == "auto":  # AST: precision comes from the probe, fp32 if missing
            chosen = "fp32"
            if probe_path.exists():
                chosen = json.loads(probe_path.read_text()).get("chosen", "fp32")
            else:
                qprint("amp_probe result missing: AST falls back to fp32")
            cfg["train"]["amp"] = chosen
            qprint(f"{exp}: amp resolved to {chosen}")
        if store is None:
            store = FeatureStore(cfg["data"]["cache"])
        for k in job["folds"]:
            if (Path(a.results_dir) / exp / f"fold{k}.json").exists():
                continue
            jlog_dir = logs / exp
            jlog_dir.mkdir(parents=True, exist_ok=True)
            jlog = open(jlog_dir / f"fold{k}.log", "a", buffering=1)

            def log(msg, _j=jlog):
                _j.write(f"[{now()}] {msg}\n")

            log(f"job start exp={exp} fold={k} commit={GIT_AT_START['commit']} "
                f"config_hash={config_hash(cfg)}")
            state["current"] = {"experiment": exp, "fold": k, "started": now(),
                                "config_hash": config_hash(cfg)}
            save_state()
            qprint(f"START {exp} fold {k}")
            t0 = time.time()
            try:
                run_round(cfg, k, a.results_dir, a.ckpt_dir, a.device, store, log=log)
                state["n_done_this_run"] += 1
                qprint(f"DONE  {exp} fold {k} in {time.time() - t0:.0f}s")
                since_commit += 1
                if since_commit >= 5:
                    since_commit = 0
                    commit_results()
            except Exception as e:
                state["n_failed_this_run"] += 1
                tb = traceback.format_exc()
                log("FAILED\n" + tb)
                with open(logs / "failures.jsonl", "a") as f:
                    f.write(json.dumps({"time": now(), "experiment": exp, "fold": k,
                                        "error": repr(e), "traceback": tb}) + "\n")
                qprint(f"FAIL  {exp} fold {k}: {e!r} (continuing)")
            finally:
                jlog.close()
                gc.collect()
            state["current"] = None
            save_state()
        commit_results()  # end of this experiment
        since_commit = 0
    state["status"] = "finished"
    state["finished"] = now()
    save_state()
    qprint("queue finished")


# Guard required: macOS multiprocessing/spawn re-imports the entry script.
if __name__ == "__main__":
    main()
