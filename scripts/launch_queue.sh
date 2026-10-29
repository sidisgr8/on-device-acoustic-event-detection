#!/bin/bash
# Start (or resume) the main queue, detached, with the Mac kept awake. Safe to re-run:
# finished folds are skipped and interrupted ones resume from their last epoch.
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
if python3 - <<'PY'
import json, os, sys
try:
    s = json.load(open("logs/queue_state.json")); os.kill(s["pid"], 0)
except Exception:
    sys.exit(1)
sys.exit(0)
PY
then echo "queue already running (see: .venv/bin/python scripts/status.py)"; exit 0; fi
nohup caffeinate -dimsu .venv/bin/python scripts/run_queue.py --queue "${1:-configs/queue_main.yaml}" --auto-commit ${QUEUE_EXTRA_ARGS:-} \
    >> logs/queue.out 2>&1 &
echo "launched; check with: .venv/bin/python scripts/status.py"
