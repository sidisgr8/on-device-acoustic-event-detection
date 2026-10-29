import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "handover_ast.py"


def load():
    spec = importlib.util.spec_from_file_location("handover_ast", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()


def test_decide_logic():
    h = load()
    probe = {"current": {"experiment": "amp_probe", "fold": 0}}
    assert h.decide({}, False, False) == "dead"
    assert h.decide({"pid": 1}, False, False) == "dead"
    assert h.decide({"pid": 1, **probe}, True, True) == "up_to_date"                 # already has the change
    assert h.decide({"pid": 1, **probe}, True, False) == "handover"
    assert h.decide({"pid": 1, "current": {"experiment": "ast", "fold": 1}}, True, False) == "too_late"
    for exp in ("cnn_baseline", "mobilenetv3_small_scratch", "mobilenetv3_small_imagenet", "crnn"):
        assert h.decide({"pid": 1, "current": {"experiment": exp, "fold": 3}}, True, False) == "wait"   # never touches CNN-family jobs
    assert h.decide({"pid": 1, "current": None}, True, False) == "wait"              # between folds


def test_is_ancestor_uses_git_history():
    h = load()
    root = git("rev-list", "--max-parents=0", "HEAD").splitlines()[-1]
    head = git("rev-parse", "HEAD")
    assert h.is_ancestor(root, head) and not h.is_ancestor(head, root) or root == head


def run_watcher(tmp_path, state, extra=(), timeout=20):
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "queue_state.json").write_text(json.dumps(state))
    return subprocess.run([sys.executable, str(SCRIPT), "--logs-dir", str(logs), "--poll", "0.2",
                           "--require-commit", git("rev-parse", "HEAD"), *extra],
                          capture_output=True, text=True, cwd=ROOT, timeout=timeout)


def test_end_to_end_handover_with_a_fake_queue(tmp_path):
    head = git("rev-parse", "HEAD")
    old_commit = git("rev-list", "--max-parents=0", "HEAD").splitlines()[-1]
    if old_commit == head:
        pytest.skip("needs a history with more than one commit")
    fake_queue = subprocess.Popen(["sleep", "300"])
    flag = tmp_path / "launched.flag"
    launch = tmp_path / "launch.sh"
    # the "fresh queue" the launcher starts: a new process that publishes a state file with the CURRENT commit
    launch.write_text(f"""#!/bin/bash
touch {flag}
sleep 300 > /dev/null 2>&1 &
NEW=$!
cat > {tmp_path}/logs/queue_state.json <<EOF2
{{"pid": $NEW, "git": {{"commit": "{head}", "dirty": false}}, "current": null, "status": "running"}}
EOF2
""")
    launch.chmod(0o755)
    state = {"pid": fake_queue.pid, "git": {"commit": old_commit, "dirty": False},
             "current": {"experiment": "amp_probe", "fold": 0}, "status": "running"}
    try:
        r = run_watcher(tmp_path, state, ["--launch-cmd", str(launch)])
        assert r.returncode == 0, r.stdout + r.stderr
        fake_queue.wait(timeout=10)
        assert fake_queue.returncode == -signal.SIGTERM                    # the stale queue was stopped
        assert flag.exists()                                               # and a fresh one was launched
        log = (tmp_path / "logs" / "handover.log").read_text()
        assert "handing over" in log and "contains required change: True" in log
    finally:
        fake_queue.kill()
        new = json.loads((tmp_path / "logs" / "queue_state.json").read_text()).get("pid")
        if new and new != fake_queue.pid:
            try:
                os.kill(new, signal.SIGKILL)
            except OSError:
                pass


def test_never_touches_a_queue_that_is_still_training_the_cnn_family(tmp_path):
    q = subprocess.Popen(["sleep", "30"])
    old_commit = git("rev-list", "--max-parents=0", "HEAD").splitlines()[-1]
    try:
        state = {"pid": q.pid, "git": {"commit": old_commit}, "current": {"experiment": "crnn", "fold": 7}}
        with pytest.raises(subprocess.TimeoutExpired):                     # keeps waiting, does nothing
            run_watcher(tmp_path, state, timeout=3)
        assert q.poll() is None                                            # the queue is still alive
    finally:
        q.kill()


def test_exits_quietly_if_running_queue_already_has_the_change(tmp_path):
    q = subprocess.Popen(["sleep", "30"])
    try:
        state = {"pid": q.pid, "git": {"commit": git("rev-parse", "HEAD")}, "current": {"experiment": "amp_probe"}}
        r = run_watcher(tmp_path, state)
        assert r.returncode == 0 and q.poll() is None
        assert "nothing to do" in (tmp_path / "logs" / "handover.log").read_text()
    finally:
        q.kill()


def test_too_late_is_reported_and_not_touched(tmp_path):
    q = subprocess.Popen(["sleep", "30"])
    old_commit = git("rev-list", "--max-parents=0", "HEAD").splitlines()[-1]
    if old_commit == git("rev-parse", "HEAD"):
        q.kill()
        pytest.skip("needs more than one commit")
    try:
        state = {"pid": q.pid, "git": {"commit": old_commit}, "current": {"experiment": "ast", "fold": 1}}
        r = run_watcher(tmp_path, state)
        assert r.returncode == 4 and q.poll() is None
        assert "TOO LATE" in (tmp_path / "logs" / "handover.log").read_text()
    finally:
        q.kill()
