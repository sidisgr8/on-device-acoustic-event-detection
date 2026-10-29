# LLM usage log

Course rule (ICT 4442 guidelines, section 9): any use of an LLM must be cited in the
report, stating exactly where it was used. This file is the running record.
Tool: Claude Code (Anthropic), model Claude Sonnet 5.5, driven by Rohan Sinha.

Each row is one file (or group of files) that Claude Code generated or substantially
edited. The human owner reviews, runs and must be able to explain every line.

## Summary for the report (plain language, grouped by area)

Tool: Claude Code (Anthropic), model Claude Sonnet 5.5, used by Rohan Sinha throughout the project. The table
below this section gives the file-level record. In every group the owner chose the decisions (listed in
`docs/DECISIONS.md` with their approval status), ran or approved the runs, and is responsible for the content.

| Area | What Claude was used for | What Claude was NOT used for |
|---|---|---|
| Data pipeline (`src/data`, `src/features`, `scripts/download_urbansound8k.py`, `build_cache.py`, `eda.py`, `db_floor_analysis.py`, `metadata_shortcut_check.py`) | Wrote the downloader, caches (waveforms, log-mel, AST fbank), fold arithmetic, normalisation, SpecAugment, EDA and the descriptive checks | Choosing the dataset, folds or front-end parameters (owner/spec decisions) |
| Models and training (`src/models`, `src/train`, `configs/*.yaml`, `scripts/train.py`, queue and launch scripts) | Wrote the CNN, MobileNetV3-Small wrapper, CRNN, AST wrapper (position-embedding crop), the training loop, the unattended job queue, the AST precision probe and the AST handover watcher | Choosing architectures, the training protocol v2 (D-26) or the AST time mask (D-36): owner decisions that Claude implemented |
| Post-training (`src/compress`, `scripts/quant_eval.py`, `prune.py`, `run_post.py`, `launch_post.sh`, `post_status.py`, queue_post configs) | Wrote the quantization routes (PyTorch FX/dynamic, ONNX Runtime), global magnitude pruning with masked fine-tuning, the 0 %-sparsity control and the extended-recipe option, the runners | Choosing sparsity levels, calibration size or the extended recipe's values (owner decisions D-31, D-33, D-43, D-45, D-47) |
| Latency (`src/benchmark/latency.py`, `scripts/benchmark_latency.py`) | Wrote the harness, the idle-machine guard and the repeat/aggregation logic | The idle-guard thresholds (owner-approved, D-41/D-42) |
| Diagnostics (`scripts/diagnose_*.py`, `report_diagnostics.py`, `results/diagnostics/`) | Wrote and ran the validation-only diagnosis of the ImageNet MobileNetV3-Small int8/pruning collapse and its plain-text report | Explaining the cause: it remains open |
| Analysis (`src/analysis`, `scripts/make_tables.py`, `make_numbers_sheet.py`, `compute_complexity.py`, `describe_models.py`) | Wrote aggregation, paired per-fold statistics (Wilcoxon, t, bootstrap, Holm), per-class/confusion/padding analysis, MACs, figures, and the generated numbers sheet | Interpreting the results in the report |
| Running the experiments | Claude Code launched, monitored and checked the training and post-training runs on the owner's machine, and reported results and anomalies | Any result is typed in by hand: all numbers are read from result files |
| Tests (`tests/`) | Wrote the unit tests (131 pass in the post environment) | |
| Documentation (`docs/PROJECT_SPEC.md`, `DECISIONS.md`, `PIPELINE.md`, `RUNBOOK.md`, `AST_PLAN.md`, `README.md`, `results/superseded_protocol_v1/README.md`) | Drafted and kept these technical documents up to date; logged the owner's decisions | The report prose, which the owner writes |
| Study material (`docs/study/*`: factsheet, viva prep, verified references, results checklist, `RESULTS_NUMBERS.md`) | Wrote study aids and a numbers sheet generated from result files; verified literature claims against source metadata | These are not report text |
| Interim report (outside the repo) | Text and literature table were drafted with an LLM, as declared in that report. Its section 4.5 numbers were not produced by experiments and are not used anywhere in this repo | |


### File-level record

| Date | File(s) | What the LLM did | Human review |
|---|---|---|---|
| 2026-10-19 | `docs/PROJECT_SPEC.md` | Drafted the consolidated spec from the course PDFs and the interim report's design sections (results excluded) | Pending |
| 2026-10-19 | `docs/DECISIONS.md` | Drafted the deviation log | Pending |
| 2026-10-19 | `scripts/download_urbansound8k.py` | Wrote the parallel, resumable Zenodo downloader with MD5 check | Pending |
| 2026-10-19 | `scripts/eda.py` | Wrote the EDA script (stats + figures) | Pending |
| 2026-10-19 | repo scaffold (first `README.md`, `requirements.txt`, `.gitignore` additions, package stubs) | Created layout | Pending |
| 2026-10-19 | `docs/DECISIONS.md` (D-12..D-19 and edits) | Logged owner-approved decisions and new pipeline decisions | Pending |
| 2026-10-19 | `src/features/params.py`, `src/features/logmel.py` | Wrote front-end parameters, log-mel, AST fbank, floor | Pending |
| 2026-10-19 | `src/data/cache.py`, `scripts/build_cache.py` | Wrote the cache builders (including the multiprocessing-spawn fix) | Pending |
| 2026-10-19 | `src/data/splits.py`, `src/data/dataset.py` | Wrote fold arithmetic, normalisation, SpecAugment, datasets | Pending |
| 2026-10-19 | `scripts/db_floor_analysis.py` | Wrote the dB-floor analysis and the selection rule | Pending |
| 2026-10-19 | `src/models/cnn.py`, `src/models/__init__.py` | Wrote the CNN baseline and model factory | Pending |
| 2026-10-19 | `src/eval/metrics.py` | Wrote metrics helpers | Pending |
| 2026-10-19 | `src/train/loop.py`, `scripts/train.py`, `configs/cnn_baseline.yaml` | Wrote the resumable training loop, CLI and config | Pending |
| 2026-10-19 | `tests/*`, `pyproject.toml`, `docs/PIPELINE.md` | Wrote the 43 unit tests, packaging and pipeline notes | Pending |
| 2026-10-19 | `src/models/mobilenet.py`, `src/models/crnn.py`, `src/models/ast.py`, `src/models/__init__.py` | Wrote the MobileNetV3-Small wrapper, CRNN, AST wrapper with position-embedding crop | Pending |
| 2026-10-19 | `src/train/queue.py`, `scripts/run_queue.py`, `scripts/status.py`, `scripts/launch_queue.sh` | Wrote the job queue, status report and launcher | Pending |
| 2026-10-19 | `src/train/loop.py`, `src/data/dataset.py` (edits) | Added traceability fields, AST data path, generalised SpecAugment axes | Pending |
| 2026-10-19 | `configs/*.yaml` (mobilenet, crnn, ast, queues) | Wrote experiment and queue configs | Pending |
| 2026-10-19 | `tests/test_models.py`, `tests/test_ast.py` | Wrote model and AST tests | Pending |
| 2026-10-19 | `docs/AST_PLAN.md`, `docs/PIPELINE.md` (edits), `docs/DECISIONS.md` (D-20..D-25) | Wrote plan and technical notes | Pending |
| 2026-10-19 | `src/train/loop.py` (protocol v2 edits), `configs/{cnn_baseline,mobilenetv3_small_*,crnn,ast}.yaml` | Implemented protocol v2 (explicit epochs/early-stopping, protocol_version in results, best.json, last.pt cleanup) | Pending |
| 2026-10-19 | `src/train/amp_probe.py`, `scripts/run_queue.py` (edits), `src/train/queue.py` (edits), `configs/queue_main.yaml`, `scripts/launch_queue.sh` | Wrote the AST precision probe, the chained queue stages, auto-commit hook | Pending |
| 2026-10-19 | `scripts/commit_results.py`, `scripts/status.py` (edits) | Wrote the finished-folds-only commit helper; status shows v2 only, stages and disk | Pending |
| 2026-10-19 | `tests/test_model_and_loop.py` (edits), `tests/test_queue_tools.py` | Updated and added tests for protocol v2, probe selection, commit helper | Pending |
| 2026-10-19 | `results/superseded_protocol_v1/README.md` | Wrote the note explaining the superseded v1 outputs | Pending |
| 2026-10-19 | `docs/DECISIONS.md` (D-22 update, D-26..D-30), `docs/PROJECT_SPEC.md`, `docs/PIPELINE.md` | Logged decisions and updated schedule/protocol text | Pending |
| 2026-10-19 | `src/analysis/{aggregate,errors,figures}.py`, `scripts/make_tables.py` | Wrote the aggregation, error-analysis and figure code | Pending |
| 2026-10-19 | `src/compress/{common,variants,prune}.py`, `scripts/quant_eval.py`, `scripts/prune.py` | Wrote quantization (PyTorch/ONNX Runtime), pruning and their drivers | Pending |
| 2026-10-19 | `src/benchmark/latency.py`, `scripts/benchmark_latency.py` | Wrote the latency harness and idle check (first version; later edits are logged below) | Pending |
| 2026-10-19 | `scripts/run_post.py`, `configs/queue_post.yaml`, `scripts/commit_results.py` (rewrite), `requirements-post.txt` | Wrote the post-training runner and extended the commit helper | Pending |
| 2026-10-19 | `tests/test_aggregate.py`, `test_figures.py`, `test_prune.py`, `test_variants.py`, `test_latency.py`, `test_queue_tools.py` (edit), `conftest.py` (edit) | Wrote tests for the new code | Pending |
| 2026-10-19 | `docs/RUNBOOK.md`, `docs/DECISIONS.md` (D-31..D-35), `docs/PIPELINE.md` (edit) | Wrote the runbook and logged decisions | Pending |
| 2026-10-19 | `docs/study/METHODS_FACTSHEET.md`, `docs/study/generated/model_summaries.md`, `scripts/describe_models.py` | Wrote the factsheet from the code/configs; wrote the script that generates layer-by-layer shapes, parameter counts and MACs | Pending |
| 2026-10-19 | `docs/study/VIVA_PREP.md` | Wrote the viva questions and answers (study material only, not report text) | Pending |
| 2026-10-19 | `docs/study/REFERENCES_VERIFIED.md` | Verified the literature claims against arXiv/Crossref metadata and PDF text; wrote the verdicts and the suggested-reference list | Pending |
| 2026-10-19 | `scripts/metadata_shortcut_check.py`, `results/eda/metadata_shortcut.json` | Wrote and ran the metadata-only shortcut check (descriptive, no model results) | Pending |
| 2026-10-19 | `tests/test_ast.py` (edit) | Added the AST patch-token-order test | Pending |
| 2026-10-19 | `docs/PROJECT_SPEC.md` (one-number correction: 233 clips at <= 32 kHz) | Corrected a miscount found while writing the factsheet | Pending |
| 2026-10-19 | `configs/ast.yaml` (time_width 40), `tests/test_specaugment.py` (edit) | Changed the AST SpecAugment time mask per the owner's decision D-36; added the tests | Pending |
| 2026-10-19 | `scripts/verify_cnn_family_unchanged.py` | Wrote the proof script (config hashes, bit-identical SpecAugment digests, changed-file list) | Pending |
| 2026-10-19 | `scripts/handover_ast.py`, `scripts/launch_queue.sh` (edit), `tests/test_handover.py` | Wrote the AST-stage handover watcher (D-37) and its tests | Pending |
| 2026-10-19 | `src/analysis/complexity.py`, `scripts/compute_complexity.py`, `results/complexity.json`, `scripts/describe_models.py` (refactor), `src/analysis/{aggregate,figures,errors}.py` and `scripts/make_tables.py` (edits), `tests/test_complexity.py`, `tests/test_aggregate.py`, `tests/test_figures.py` (edits) | Added MACs to the master table, accuracy-vs-MACs and accuracy-vs-latency figures, top-confusions table | Pending |
| 2026-10-19 | `docs/study/RESULTS_CHECKLIST.md`, `docs/DECISIONS.md` (D-36, D-37), `docs/PROJECT_SPEC.md`, `docs/RUNBOOK.md`, `docs/study/METHODS_FACTSHEET.md`, `docs/study/VIVA_PREP.md` (edits) | Wrote the checklist (pointers only); logged decisions and updated study material | Pending |
| 2026-10-20 | `src/analysis/{paired,curves}.py`, `src/analysis/{aggregate,figures}.py` (edits), `scripts/make_tables.py` (edit), `tests/test_paired.py`, `tests/test_figures.py` (edit) | Wrote the paired per-fold analysis (Wilcoxon, paired t, bootstrap CI, Holm), training-curve and accuracy-vs-size figures, the ablation rows/table and their tests | Pending |
| 2026-10-20 | `configs/ablation_a1.yaml`, `configs/queue_ablation.yaml`, `configs/queue_post_run1.yaml`, `scripts/launch_post.sh`, `scripts/post_status.py`, `scripts/quant_eval.py` (edit: per-fold failure recording, check-before-evaluate), `scripts/status.py` (edit: sample std), `docs/DECISIONS.md` (D-38, D-39), `docs/RUNBOOK.md`, `docs/study/RESULTS_CHECKLIST.md` (edits) | Declared the ablation before its results, wrote the post-training run launcher/status and logged the decisions | Pending |
| 2026-10-20 | `src/benchmark/latency.py` (repeats, aggregation, machine state, ancestor-aware idle check, load threshold per D-41/D-42), `scripts/benchmark_latency.py` (3 shuffled repeats, one pruned variant), `src/analysis/aggregate.py` (latency table spread column), `scripts/make_tables.py` (edit), `tests/test_latency.py` (edit), `docs/DECISIONS.md` (D-40..D-42), `docs/RUNBOOK.md` | Extended the latency harness to the owner's protocol, logged the decisions | Pending |
| 2026-10-20 | `scripts/make_numbers_sheet.py`, `docs/study/RESULTS_NUMBERS.md` (generated, no prose), `docs/study/RESULTS_CHECKLIST.md` (edit) | Wrote the generator of the final-numbers facts sheet from the result files | Pending |
| 2026-10-20 | `scripts/diagnose_mnv3_quant.py`, `scripts/diagnose_mnv3_quant2.py`, `scripts/diagnose_mnv3_quant3.py`, `scripts/diagnose_prune_recovery.py`, `scripts/report_diagnostics.py`, `results/diagnostics/*.json`, `scripts/make_numbers_sheet.py` (edit: diagnostics blocks), `configs/queue_post_run2.yaml`, `docs/DECISIONS.md` (D-44 to D-46) | Wrote the validation-only diagnostics of the ImageNet MobileNetV3-S int8 and pruning collapse, the plain-text report script and the 0 %-sparsity control stage | Pending |
| 2026-10-20 | `scripts/prune.py` (edit: `--ft-lr`, `--ft-epochs`, `--dir-suffix`, `recipe` field), `src/analysis/aggregate.py`, `src/analysis/figures.py` (edits: extended-recipe rows labelled and separate), `scripts/report_diagnostics.py` (edit), `configs/queue_post_run3.yaml`, `tests/test_aggregate.py` (edit), `docs/DECISIONS.md` (D-47) | Implemented the owner-approved extended pruning fine-tune recipe as additional labelled rows | Pending |
| 2026-10-20 | `README.md` (rewritten: summary, headline table copied from `results/tables`, setup, reproduction commands, limitations), `docs/LLM_USAGE.md` (summary section) | Wrote the final README and this summary | Pending |

The files in `docs/study/` are study aids for the owner and **are not report text**; the report prose is written by the owner.

Also outside the repo: the interim report (25 Sep) text and literature table were
drafted with an LLM, as already declared in that report. Its section 4.5 numbers
were not produced by experiments and must not be used (see `docs/DECISIONS.md`).
