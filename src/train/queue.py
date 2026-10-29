"""Shared helpers for the job queue and the status report."""
import json
import os
from pathlib import Path

import yaml


def load_queue(path):
    q = yaml.safe_load(open(path))
    jobs = []
    for j in q["jobs"]:
        cfg = yaml.safe_load(open(j["config"]))
        if j.get("stage"):  # non-training stage such as the AST precision probe
            jobs.append({"stage": j["stage"], "config_path": j["config"], "cfg": cfg})
            continue
        jobs.append({"config_path": j["config"], "experiment": cfg["experiment"], "cfg": cfg,
                     "folds": j.get("folds", q["folds"]), "est_epoch_s": j.get("est_epoch_s")})
    return jobs


def pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def read_state(logs_dir) -> dict:
    p = Path(logs_dir) / "queue_state.json"
    return json.loads(p.read_text()) if p.exists() else {}
