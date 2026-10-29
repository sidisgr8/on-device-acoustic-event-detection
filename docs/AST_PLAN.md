# AST plan (for review; nothing here has been launched)

**Model.** `MIT/ast-finetuned-audioset-10-10-0.4593` (HuggingFace), 85,625,098 parameters with the new
10-way head. Input is the cached, checkpoint-normalised Kaldi fbank, 400 x 128. Code: `src/models/ast.py`,
config: `configs/ast.yaml`, queue: `configs/queue_ast.yaml`.

**Position-embedding crop (D-08), what was done and the shapes.**
Checkpoint grid 12 (freq) x 101 (time) patches + 2 special tokens = (1, 1214, 768). For 400 frames the patch
grid is 12 x 39, so the target is (1, 470, 768). All 12 frequency rows are kept; time columns [31, 70) are
taken (a *centred* crop: 50 - 39//2 = 31), the 2 special-token embeddings are prepended. Unit tests check
shapes, that the cropped values equal the checkpoint slice, that the other 200 pretrained tensors are loaded
bit-identically and that only the 10-way head is new.
**Discrepancy to confirm:** the approval text said "take the first N time positions", but the AST authors'
code (YuanGongND/ast, `ast_models.py`, audioset branch) centre-crops the time axis. I implemented the authors'
centred crop, since "the standard approach from the authors' code" was the primary instruction. `pos_crop: first`
in the config switches to the first-N variant. It must be chosen on principle, not on test results.

**Training protocol (spec 4.4).** AdamW lr 5e-5 (wd 0.01), cosine decay to 0 over the 5 epochs, no warm-up,
batch 16 (435 steps/epoch), label smoothing 0.1, SpecAugment as for the other models (applied on the
(T, F) layout), seed 42, best-validation-accuracy epoch kept, test fold evaluated once. Early stopping is
disabled (patience = epochs).

**Folds and fallback.** Order 1, 5, 9, 2, 3, 4, 6, 7, 8, 10. If time runs short, stop after the finished folds
and label AST rows "n folds" everywhere.

**Compute estimate** (from the probe, to be replaced by the first real fold): ~68 ms/clip training in fp16
-> ~8 min/epoch, ~40 min/fold, ~6.7 h for 10 folds. Needs ~9 GB of GPU memory at batch 16, the GPU to itself,
and `caffeinate`.

**Open item before launch (D-11).** fp16 (with GradScaler) vs bf16 vs fp32 for AST. The config says fp16
provisionally. Proposed check, run once the CNN queue is idle: ~30 optimiser steps on fold-1 *training* data
in each mode, comparing loss finiteness and step time (no validation or test data used). Pick the fastest
mode with finite, matching loss.

**After AST (not launched).** Quantization (PyTorch dynamic int8 as measured, and ONNX Runtime int8), pruning at
30 % and 70 % (3 epochs fine-tune, folds 1/5/9), then the latency/size benchmark on an idle machine.
