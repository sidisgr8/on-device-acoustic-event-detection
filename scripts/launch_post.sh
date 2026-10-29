#!/bin/bash
# Start (or resume) the first post-training run, detached, with the Mac kept awake:
# A-1 ablation -> pruning -> quantization eval -> tables (configs/queue_post_run1.yaml). Latency is NOT included.
# Safe to re-run after an interruption: finished folds/variants are skipped, a test fold is never evaluated twice.
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs/post
if [ -f logs/post/run_post.pid ] && kill -0 "$(cat logs/post/run_post.pid)" 2>/dev/null; then
    echo "post run already running (pid $(cat logs/post/run_post.pid)); see: .venv-post/bin/python scripts/post_status.py"; exit 0
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "git tree has uncommitted changes: commit first so every result records a clean commit"; exit 1
fi
nohup caffeinate -dimsu .venv-post/bin/python scripts/run_post.py --queue "${1:-configs/queue_post_run1.yaml}" \
    >> logs/post/run1.out 2>&1 &
echo $! > logs/post/run_post.pid
echo "launched pid $(cat logs/post/run_post.pid); check with: .venv-post/bin/python scripts/post_status.py"
