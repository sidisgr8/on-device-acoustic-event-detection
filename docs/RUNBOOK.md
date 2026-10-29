# Runbook: what to run once the training queue has finished

Everything below is for the **post-training** phase. Interpreter for all of it: `.venv-post/bin/python`
(same pinned torch as training, plus onnx/onnxruntime; created from `requirements-post.txt`, kept separate so the
training venv is never modified under a running queue, D-32).

## 0. Preconditions (check before anything else)

```bash
.venv/bin/python scripts/status.py          # CNN family 10/10 each, AST 10/10 (or n/10, see AST fallback), queue not running
ps -axo pid,command | grep -E "run_queue|run_post|train.py|prune.py|quant_eval" | grep -v grep   # must print nothing
git status --short                          # finished results are auto-committed; only new files from stages below appear
```

If a fold failed, `status.py` shows it; fix the cause and run `scripts/launch_queue.sh` again before continuing.

## 0b. AST handover watcher (D-37): start it once, before the CNN family finishes

The training queue is one long-lived process with the old AST config and old commit in memory. `scripts/handover_ast.py` swaps it for a fresh
process at the CNN-family/AST boundary (it never touches a queue that is training a CNN-family fold):

```bash
nohup nice -n 19 .venv/bin/python scripts/handover_ast.py >> logs/handover.out 2>&1 &
tail logs/handover.log            # "watcher start ..." now; "handing over" + "new queue ... contains required change: True" at ~01:45
```

After the AST stage has started, verify (no test data involved):

```bash
.venv/bin/python - <<'PY'
import json; r = json.load(open("results/ast/fold1.json"))      # exists after AST fold 1 finishes
print(r["git"], r["config"]["specaugment"], r["config_hash"])      # expect time_width 40 and the commit that contains D-36
PY
```

If the watcher is not running or died before the boundary: stop the queue when `status.py` shows `[stage] amp_probe` running, then `scripts/launch_queue.sh`.
If `logs/handover.log` says TOO LATE: an AST fold is running with the stale config; stop the queue, move `checkpoints/ast`, `results/ast` aside, run `scripts/launch_queue.sh`.

## 1. Quantization and pruning (automatic, one command)

```bash
caffeinate -dimsu nohup .venv-post/bin/python scripts/run_post.py > logs/post.out 2>&1 &
tail -f logs/post.out                       # or: ls logs/post/ ; tail logs/post/quant_eval.log
```

Runs, in order (`configs/queue_post.yaml`, `.venv-post/bin/python scripts/run_post.py --list` shows it):

| stage | needs | writes | rough time (estimate, not measured) |
|---|---|---|---|
| `quant_eval` | CPU only | `results/quant/<exp>/<variant>/fold<k>.json`; artifacts in `checkpoints/compressed/` (gitignored) | CNN family ~35 min; AST ~2.5 h (PyTorch dynamic int8 AST is ~7x slower than fp32 on this Mac, a finding) |
| `prune` | GPU (MPS) | `results/prune/<exp>/s<30\|50\|70>/fold<k>.json`; `checkpoints/pruned/` (gitignored) | CNN family ~40 min; AST ~2.5 h |
| `tables` | CPU, seconds | `results/tables/*.csv|md`, `results/figures/*.png` | seconds |

* Variants per model: `fp32_torch`, PyTorch int8 (`torch_static_int8_fx` for CNN/MobileNetV3, `torch_dynamic_int8` for CRNN/AST),
  `onnx_fp32` (checked against PyTorch to 1e-3 before quantizing), `onnx_int8` (ORT static QDQ for conv models, dynamic for CRNN/AST).
* Pruning: global magnitude, 30/50/70 % (AST 30/70 %), 3 fine-tune epochs, folds 1/5/9.
* Both stages are idempotent: finished (model, fold, variant) files are skipped, so you can simply re-run the same command.
* After each stage `scripts/commit_results.py` commits only finished test-split result files, then tables/figures.
* A stage that fails is recorded in `logs/post/failures.jsonl` (log: `logs/post/<stage>.log`) and the next stage still runs.
  A **variant** that cannot be built/exported (e.g. an ONNX export problem) is stored as `"error": ...` in its result file and
  shown as "failed" in the tables: that is a finding, not something to retry in a loop. To retry one anyway, delete its JSON.
  A pruning run that fails leaves `fold<k>.error.txt` and no result file, so re-running retries it.
* The test fold is evaluated once per (variant, fold) / (sparsity, fold): result files are created in exclusive mode.

## 1b. Post-training run 1 (A-1 ablation, pruning, quantization, tables): one launcher

```bash
scripts/launch_post.sh                               # detached, caffeinate; refuses if the tree has uncommitted tracked changes
.venv-post/bin/python scripts/post_status.py         # stage, counts of results (A-1 x/10, prune x/42, quant x/200), failures
tail -f logs/post/run1.out                           # stage log; per-stage logs in logs/post/<stage>.log
```

Order: `ablation_a1` (GPU) -> `prune` (GPU, folds 1/5/9) -> `quant_eval` (CPU, test split, all folds) -> `tables`. Latency is not part of it.
Recovery after an interruption (sleep, reboot, crash): run `scripts/launch_post.sh` again; finished folds/variants are skipped, a test fold is
never evaluated twice (result files are opened with mode `x`). A failing item is recorded (`*.error.txt` next to the results, or
`logs/failures.jsonl` / `logs/post/failures.jsonl`) and skipped; fix the cause and relaunch to retry it.
Stop cleanly: `kill $(cat logs/post/run_post.pid)` (the running stage is interrupted; relaunch resumes).

## 2. Latency (MANUAL, needs an idle machine and your go-ahead)

Only after step 1 has fully finished and nothing else is running. Wait ~5 minutes so the 1-minute load average settles.

```bash
.venv-post/bin/python scripts/benchmark_latency.py             # or: scripts/run_post.py --stage latency (same thing)
```

* Repeated 3 times with a different model order each time (given, reversed, rotated); the reported number is the median
  across repeats of each repeat's median/p95, with the spread across repeats and a flag above 10 % (`noisy`).
  Fold-1 weights only and one pruned variant (70 %) per model (D-40). Machine state (load, `pmset` power and thermal) is
  recorded before/after each repeat. Expect ~40-50 minutes in total; the AST PyTorch dynamic-int8 rows dominate.

* It checks `ps` for any training/post job and the 1-minute load average (must be <= 2.0, D-41, D-42; the launch procedure also waits until it stays below that for ~3 minutes) and **refuses** (exit 2) otherwise;
  there is no override. Close other apps; do not touch the machine while it runs.
* Batch 1, one 4 s validation clip, 20 warm-up + 200 timed runs, median and p95, 1 and 4 threads, forward pass and
  feature extraction timed separately, for every model, its fp32 form, each quantized variant and the 70 % pruned model.
* Writes `results/latency/latency.json` including machine metadata (CPU, 4P+6E core split, torch/ORT versions, thread
  settings, date, load average) and the thread-count caveat. If it fails halfway nothing is written; just re-run.
* Then refresh the tables and commit:

```bash
.venv-post/bin/python scripts/run_post.py --stage tables
```

## 3. Final checks

```bash
.venv-post/bin/python -m pytest -q                 # 97+ tests
git status --short                                 # should be clean; git log --oneline | head
cat results/tables/tables.md
```

## Dry runs and correctness checks (no test fold, no timing)

```bash
export OMP_NUM_THREADS=1
nice -n 19 .venv-post/bin/python scripts/quant_eval.py --experiments cnn_baseline --folds 1 --split val \
    --threads 1 --out /tmp/q --artifacts /tmp/qa                     # --split val is the default
nice -n 19 .venv-post/bin/python scripts/prune.py --dry-run --experiments cnn_baseline --folds 1 \
    --sparsities 0.3 --out /tmp/p --artifacts /tmp/pa                # CPU, 3 batches of validation data
```

## If something goes wrong

* **Mac slept / process died:** re-run the same command; every stage skips finished work.
* **`run_post.py` refuses to start:** a job is still running; `ps` shows which one.
* **A stage log shows a traceback:** read `logs/post/<stage>.log`, fix, re-run that stage with `--stage <name>`.
* **Disk:** `best.pt` (~3.6 GB), compressed artifacts (AST ONNX files ~0.5 GB per fold) and pruned weights are all gitignored;
  `status.py` prints free space.
