# Superseded: training protocol v1 (do not use in any table or plot)

These files come from the first, aborted launch of the CNN baseline on 2026-10-19 (git commit
`f37e42a`, ~21:18 IST), run under **protocol v1**: up to 40 epochs, cosine schedule defined over
40 epochs, early stopping on validation accuracy with patience 8.

* `cnn_baseline/fold1.json`, `fold1_preds.csv`, `fold1_epochs.jsonl`: a complete round 1
  (stopped at epoch 14; learning rate had barely decayed).
* `cnn_baseline/fold2_epochs.jsonl`: **partial**, the queue was stopped at about epoch 11 of fold 2;
  there is no result file for fold 2.

**Why superseded.** The owner replaced the protocol for all CNN-family models (CNN baseline,
MobileNetV3-Small scratch and ImageNet, CRNN) with protocol v2: a fixed 30 epochs, 2 warm-up epochs then
cosine decay over exactly those 30 epochs, no early stopping, best-validation-accuracy checkpoint
evaluated once on the test fold. The reason is mechanical: with patience 8 the run ended before the
cosine schedule could decay the learning rate, and validation accuracy was noisy epoch to epoch. See
`docs/DECISIONS.md` D-26. The fold-1 test accuracy of this run had been seen before that decision;
the decision rests on the schedule mechanics and the validation curves, not on it.

Kept for the record only. `scripts/status.py`, the result tables and all figures read protocol v2
results (`protocol_version: 2` in each result file) and ignore this directory. The matching
checkpoints and logs were moved to `checkpoints/superseded_protocol_v1/` and `logs/superseded_protocol_v1/`
(untracked).
