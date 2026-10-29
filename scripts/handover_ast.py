"""One-shot handover of the training queue at the CNN-family -> AST boundary.

Why: the queue is ONE long-lived process. It loaded its code and configs into memory at start and stamped its
results with the git commit of that moment, so config/code changes committed afterwards (here: AST SpecAugment
time mask 40 frames, D-36) are invisible to its AST stage. This watcher leaves the CNN-family folds completely
alone. Only when the running queue enters the `amp_probe` stage (i.e. all CNN-family jobs are done) and its
recorded commit does not contain the required change, it stops that process (nothing but a few-minute probe is
lost; all finished work is skipped on restart) and starts a fresh queue from the current commit. The fresh queue
re-reads every config and records the current HEAD for the AST results.

    nohup nice -n 19 python scripts/handover_ast.py >> logs/handover.out 2>&1 &

Decisions (see `decide`): wait / up_to_date (running queue already has the change: nothing to do) /
handover / too_late (an AST fold already runs with the stale config: do NOT touch it, shout) / dead.
Log: logs/handover.log.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

HANDOVER_STAGE = "amp_probe"


def now():
    return datetime.now().isoformat(timespec="seconds")


def git(*args, check=True):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=check)


def is_ancestor(required: str, running: str) -> bool:
    """True if commit `required` is contained in the history of commit `running`."""
    return git("merge-base", "--is-ancestor", required, running, check=False).returncode == 0


def pid_alive(pid) -> bool:
    """Alive = signalable and not a zombie (a terminated child that nobody has reaped yet still accepts signal 0)."""
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    stat = subprocess.run(["ps", "-o", "stat=", "-p", str(int(pid))], capture_output=True, text=True).stdout.strip()
    return bool(stat) and not stat.startswith("Z")


def decide(state: dict, alive: bool, has_change: bool) -> str:
    """Pure decision logic. state = logs/queue_state.json; has_change = running queue's commit includes the change."""
    if not state or not alive:
        return "dead"
    if has_change:
        return "up_to_date"
    cur = state.get("current") or {}
    if cur.get("experiment") == HANDOVER_STAGE:
        return "handover"
    if cur.get("experiment") == "ast":
        return "too_late"
    return "wait"


def code_tree_dirty() -> bool:
    out = git("status", "--porcelain", "--", ".", ":(exclude)results", ":(exclude)logs").stdout.strip()
    return bool(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--require-commit", default=None,
                    help="commit that must be in the queue's history (default: last commit touching configs/ast.yaml)")
    ap.add_argument("--launch-cmd", default="scripts/launch_queue.sh", help="command that starts a fresh queue")
    ap.add_argument("--timeout-h", type=float, default=24.0)
    a = ap.parse_args()
    logs = Path(a.logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    logf = open(logs / "handover.log", "a", buffering=1)

    def log(msg):
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")

    required = a.require_commit or git("log", "-1", "--format=%H", "--", "configs/ast.yaml").stdout.strip()
    spec = yaml.safe_load(open("configs/ast.yaml"))["specaugment"]
    log(f"watcher start pid={os.getpid()}; required commit {required[:10]}; configs/ast.yaml specaugment={spec}")
    if not a.require_commit and git("diff", "--quiet", "HEAD", "--", "configs/ast.yaml", check=False).returncode != 0:
        log("ABORT: configs/ast.yaml has uncommitted changes; commit it first so the AST results record the right commit")
        sys.exit(3)
    deadline = time.time() + a.timeout_h * 3600
    state_path = logs / "queue_state.json"
    while time.time() < deadline:
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        alive = pid_alive(state.get("pid"))
        qc = (state.get("git") or {}).get("commit")
        has_change = bool(qc) and is_ancestor(required, qc)
        verdict = decide(state, alive, has_change)
        if verdict == "wait":
            time.sleep(a.poll)
            continue
        if verdict == "up_to_date":
            log("running queue already contains the change: nothing to do")
            return
        if verdict == "dead":
            log("no live queue: nothing to hand over (a fresh launch reads the current configs anyway)")
            sys.exit(2)
        if verdict == "too_late":
            log("TOO LATE: the running queue is already training AST with the OLD config. NOT touching it. "
                "Stop it manually, move checkpoints/ast + results/ast aside, and relaunch scripts/launch_queue.sh")
            sys.exit(4)
        # verdict == "handover"
        pid = int(state["pid"])
        log(f"queue pid {pid} entered stage '{HANDOVER_STAGE}' with commit {str(state['git']['commit'])[:10]} "
            f"(lacks {required[:10]}): handing over")
        os.kill(pid, signal.SIGTERM)
        for _ in range(60):
            if not pid_alive(pid):
                break
            time.sleep(0.5)
        else:
            log("queue ignored SIGTERM for 30 s: SIGKILL")
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
        env = dict(os.environ)
        if code_tree_dirty():
            env["QUEUE_EXTRA_ARGS"] = "--allow-dirty"
            log("WARNING: code tree has uncommitted changes: launching with --allow-dirty; results will record dirty=true")
        subprocess.run([a.launch_cmd], env=env, check=False)
        for _ in range(120):
            time.sleep(1)
            st = json.loads(state_path.read_text()) if state_path.exists() else {}
            if st.get("pid") != pid and pid_alive(st.get("pid")):
                c = (st.get("git") or {}).get("commit")
                log(f"new queue pid {st['pid']} running, commit {str(c)[:10]}, contains required change: "
                    f"{is_ancestor(required, c) if c else 'unknown'}")
                return
        log("ERROR: new queue did not appear within 120 s; run scripts/launch_queue.sh manually")
        sys.exit(5)
    log("timeout: no handover needed within the watch window")


if __name__ == "__main__":
    main()
