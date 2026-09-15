# On-Device Acoustic Event Detection (ICT 4442 Deep Learning Mini Project)

Comparison of a CNN, a CRNN and an Audio Spectrogram Transformer on UrbanSound8K under one
protocol, with an accuracy-vs-efficiency focus. The frozen design is in
`docs/Interim_Report_PartB.pdf` (Sections 3 and 4). This repo holds the shared pipeline and the
first models; AST and the compression experiments come later.

## What is here

| Path | What it does | Owner |
|---|---|---|
| `scripts/download_data.py` | downloads and verifies UrbanSound8K (about 6 GB) | shared |
| `src/data.py`, `scripts/build_cache.py` | 16 kHz / 4 s / 128x201 log-mel cache, fold splits, train-only normalisation, SpecAugment | Rohan |
| `src/models/cnn.py`, `mobilenet.py` | CNN baseline, MobileNetV3-Small (scratch or ImageNet) | Rohan |
| `src/models/crnn.py` | CNN + bidirectional GRU | Akshdeep |
| `src/train.py` | cross-validation training / evaluation loop | shared |
| `src/bench.py` | parameters, model size, CPU latency | Rohan |
| `src/aggregate.py` | builds the report table and confusion matrices from real results | shared |
| `tests/` | split-leakage, normalisation and shape tests | shared |
| `docs/` | guidelines, synopsis, interim report (PDF and Word) | - |

Not included yet: AST fine-tuning (Siddharth), quantization and pruning, EDA plots.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q          # should pass before you do anything else
```

## Run

```bash
python scripts/download_data.py                  # data/UrbanSound8K/  (checks MD5 and 8,732 files)
python scripts/build_cache.py                    # cache/X.npy, y.npy, fold.npy

# quick preliminary run: 3 folds, 8 epochs (use --device cuda or cpu elsewhere)
python -m src.train --model cnn  --folds 1,5,9 --epochs 8 --patience 3 --device mps
python -m src.train --model crnn --folds 1,5,9 --epochs 8 --patience 3 --device mps

python -m src.bench --model cnn                  # use the SAME machine for every model
python -m src.bench --model crnn
python -m src.aggregate --models cnn,crnn        # -> results/summary.md + confusion matrices
```

Other model names: `mobilenet_scratch`, `mobilenet_pretrained`. The full protocol is
`--folds 1,2,3,4,5,6,7,8,9,10 --epochs 40 --patience 8`.

## Rules the code follows

- Split for round k: test = fold k, validation = fold (k mod 10) + 1, train = the other 8.
- Normalisation statistics come from the training folds only.
- The best-validation epoch is chosen first; the test fold is evaluated once.
- `aggregate.py` has no default values: a table can only contain numbers from real runs.
- Always state exactly which folds and epoch cap produced a number.

## Honest status

This code was written with an LLM (Claude). It has only been syntax-checked: it has NOT been
run, not even on fake data, and not on the real dataset. Run `python -m pytest tests -q`
first, and expect to fix small things (paths, library versions, MPS operator quirks, the
Zenodo URL) when you first run it. Each member must be able to explain their
own model in the viva, so read your file and the Report Section 4 entry before relying on it.
Declare LLM use in the report's integrity statement.
