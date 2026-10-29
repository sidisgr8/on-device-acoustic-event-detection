# Shared pipeline: how it works and how to run it

## Build the caches (once, ~20 s)

```bash
python scripts/build_cache.py            # waveforms -> raw log-mel -> AST fbank
python scripts/db_floor_analysis.py      # evidence for the -80 dB floor (D-15)
```

`data/cache/` (gitignored, ~4.7 GB), all arrays row-aligned with `index.csv`, which has the row
order of `metadata/UrbanSound8K.csv`:

| file | shape | content |
|---|---|---|
| `waveforms.npy` | (8732, 64000) f32 | mono, 16 kHz (librosa soxr_hq), zero-padded / cropped to 4.0 s |
| `logmel_raw.npy` | (8732, 128, 201) f32 | power dB re 1.0, `amin=1e-10`, **no floor** (padding = -100 dB) |
| `ast_fbank.npy` | (8732, 400, 128) f32 | un-normalised Kaldi fbank as in `ASTFeatureExtractor` |
| `index.csv` | 8732 rows | metadata plus `orig_sr`, `orig_samples`, `n_samples_16k`, `padded`, `cropped`, `n_valid_frames`, `peak` |

1,399 clips are padded and 8 are cropped (matches the EDA).

## Splits, normalisation, augmentation

* `src/data/splits.py`: round k -> test = fold k, val = fold (k mod 10)+1, train = the other 8.
* `src/data/dataset.py`: `make_round` applies the -80 dB floor, computes scalar mean/std on the
  training folds (real frames only), and returns normalised train/val/test `BatchDataset`s.
* `SpecAugment` (2 freq masks <= 16 bins, 2 time masks <= 20 frames) is a module that is the
  identity unless `.train()`; the loop only applies it to training batches.

## Train

```bash
python scripts/train.py --config configs/cnn_baseline.yaml --folds 1 5 9     # or all 10
caffeinate -i python scripts/train.py --config ... &                          # unattended
```

**Protocol v2** (D-26): every config declares `protocol_version: 2` and an explicit
`train.early_stopping` (the loop refuses to run otherwise). CNN family: 30 fixed epochs, 2 warm-up epochs then
cosine over exactly 30, best-val-accuracy checkpoint tested once.

Per round it writes `results/<experiment>/fold<k>.json` (accuracy, macro-F1, confusion matrix,
per-class F1, accuracy by padded/full, train time, per-epoch history, config, git commit),
`fold<k>_epochs.jsonl` and `fold<k>_preds.csv`. Checkpoints (`best.pt`, `last.pt`) go to
`checkpoints/<exp>/fold<k>/` (gitignored): `best.pt` + `best.json` are kept forever (needed by the
quantization/pruning/latency steps), `last.pt` is removed once the fold has finished. A finished fold is skipped (the result file is opened with mode `x`,
so the test fold cannot be evaluated twice); an interrupted fold resumes from `last.pt`.

## Queue (unattended runs)

```bash
scripts/launch_queue.sh                      # start or resume (detached, caffeinate -dimsu, auto-commit)
.venv/bin/python scripts/status.py           # done / running / pending / failed, mean acc, ETA, disk
```

`scripts/run_queue.py` runs `configs/queue_main.yaml` sequentially: skips finished folds, resumes
interrupted ones from `last.pt`, writes `logs/<experiment>/fold<k>.log` per job, records failures in
`logs/failures.jsonl` and carries on. Re-running the launcher retries failed folds. `logs/` is gitignored.

The queue is: CNN baseline, MobileNetV3-S scratch, MobileNetV3-S ImageNet, CRNN (10 folds each), then the AST
chain (D-27): an `amp_probe` stage (fp32/fp16/bf16, 30 steps on fold-1 training clips) that picks the AST
precision by a pre-fixed rule and appends D-28 to `docs/DECISIONS.md`, then AST folds 1,5,9,2,3,4,6,7,8,10.
Quantization, pruning and latency benchmarks are **not** in the queue.

`scripts/commit_results.py` commits only finished folds (never partial epoch logs); the queue calls it every
5 completed folds and at the end of each experiment. `results/superseded_protocol_v1/` holds the abandoned
protocol-v1 outputs; nothing reads it.

## Post-training tools (after the queue; see `docs/RUNBOOK.md` for the order)

All run in `.venv-post` (D-32). Nothing here is part of the training queue.

| tool | purpose |
|---|---|
| `scripts/quant_eval.py` | int8 variants (PyTorch + ONNX Runtime), accuracy / macro-F1 / size per variant per fold, no retraining |
| `scripts/prune.py` | global magnitude pruning + 3-epoch masked fine-tune, folds 1/5/9 |
| `scripts/benchmark_latency.py` | latency/efficiency harness; refuses unless the machine is idle (manual) |
| `scripts/run_post.py` + `configs/queue_post.yaml` | runs quant_eval, prune, tables in order; latency only on request |
| `scripts/make_tables.py` | comparison tables (CSV + Markdown) and figures from result files, v2 only |
| `scripts/commit_results.py` | commits only finished, final result files (training folds, compression, tables/figures) |
| `scripts/compute_complexity.py` | parameters and MACs per clip for every model from the model code -> `results/complexity.json` (feeds the MACs column and figure) |
| `scripts/handover_ast.py` | one-shot watcher that restarts the queue with the current config/commit at the CNN-family to AST boundary (D-37) |
| `scripts/verify_cnn_family_unchanged.py` | proves a change did not alter CNN-family training (config hashes, bit-identical SpecAugment, changed files) |

Code: `src/compress/{common,variants,prune}.py`, `src/benchmark/latency.py`, `src/analysis/{aggregate,errors,figures}.py`.

## Tests

```bash
pytest -q          # 97 tests in .venv-post (the ONNX/quantization tests skip in the training venv): split arithmetic, no leakage, train-only stats, SpecAugment, cache shapes,
                   # AST front-end vs HuggingFace, model shape, loop end-to-end and resume
```

## Smoke test (not a result)

The CNN smoke runs wrote to a scratch directory outside the repo. Their numbers are not findings
and are not used for any decision except timing and AMP stability (D-11). Measured on the M5
(MPS, fold 1, 6,971 training clips, batch 64, 108 steps/epoch): about **14.3 s/epoch fp32**,
13.0 s with fp16/bf16; the data path is ~2 ms/step of ~120 ms, so the model dominates.
