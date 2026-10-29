# Project specification (consolidated)

Sources: `ICT_4442_Mini_Project_Guidelines.pdf`, `DL_Synopsis (1).pdf`, and the **design
sections** of `DL_Interim_Report (1).docx` (sections 3, 4.1-4.4, 4.6).

> **Excluded on purpose:** section 4.5 of the interim report (the "preliminary results"
> table). Those numbers were never produced by experiments. They appear nowhere in this
> repo, and the interim's literature claims are treated as unverified until checked
> against the original papers. Anything numeric below that is *not* a spec value comes
> from a measurement made on this machine and says so.

## 1. Task and dataset

* **Task:** 10-class single-label urban sound classification from <=4 s clips, studied as
  an accuracy vs efficiency trade-off for on-device use.
* **Dataset:** UrbanSound8K (Salamon et al. 2014), Zenodo record 1203745 (CC BY-NC 4.0).
  8,732 WAV clips, 10 classes, 10 predefined folds, `metadata/UrbanSound8K.csv`.
* **Classes:** air_conditioner, car_horn, children_playing, dog_bark, drilling,
  engine_idling, gun_shot, jackhammer, siren, street_music.
* Folds are grouped by source recording; they are **never re-split**.

## 2. Split protocol (from interim section 3)

For round k = 1..10: test = fold k; validation = fold (k mod 10)+1; train = the other 8
folds. Model selection = best validation-accuracy epoch; the test fold is evaluated once
per round. Report per-fold metrics as mean +/- std over the 10 test folds, plus a
confusion matrix summed over folds.

## 3. Preprocessing (shared by all models)

1. Load mono, resample to 16 kHz.
2. Zero-pad / crop to 4.0 s (64,000 samples).
3. CNN / CRNN / MobileNetV3 input: log-mel, power->dB, n_fft 1024, hop 320, 128 mels,
   20-8000 Hz, giving 128 x 201 (mel x frames), clamped below at **-80 dB** (D-15).
4. AST input: the AST-standard Kaldi fbank (128 bins, 10 ms shift) for 400 frames, with the
   checkpoint's own normalisation. The only place the front-end differs; reported as such.
5. Normalisation (CNN family): scalar mean/std from the *training folds of that round* only, over real
   (non-padded) frames only (D-16).
6. Augmentation (train batches only): SpecAugment, 2 frequency masks (<=16 bins), 2 time
   masks of up to **400 ms**: 20 frames for the 20 ms-hop log-mel models, **40 frames for AST** (10 ms shift), so the
   augmentation is identical in seconds (D-36, owner-approved before any AST result). No waveform augmentation.
7. Waveforms and log-mels are cached once on disk (gitignored) and shared by all runs.

## 4. Models

| # | Model | Pretrained | Notes |
|---|---|---|---|
| 1a | CNN baseline: 4 x (conv3x3-BN-ReLU-maxpool), 32/64/128/256 ch, GAP, dropout 0.3, linear | no | simple reference |
| 1b | MobileNetV3-Small, 1-channel input, 10-way head | no | `timm`, in_chans=1 |
| 1c | MobileNetV3-Small, ImageNet init, lr 3e-4 | ImageNet | pretraining control |
| 2 | CRNN: 3 conv blocks (32/64/128; pool (2,2),(2,2),(2,1)) -> 2048-d per step -> linear 128 -> 1-layer BiGRU(128) -> mean+max over time -> dropout 0.3 -> linear | no | GRU over LSTM for params; LSTM = optional ablation |
| 3 | AST fine-tuned, 400-frame input, 10-way head | ImageNet + AudioSet | ~85.6 M params |

Fallback for 1b: EfficientNet-Lite0, decided on *validation* accuracy only.

## 5. Common training protocol (interim 4.1)

CE with label smoothing 0.1; AdamW, wd 0.01; lr 1e-3 (from scratch) with 2 warm-up epochs
then cosine; batch 64; **protocol v2 (D-26): fixed 30 epochs, cosine over exactly 30, no early stopping,
best-val-accuracy checkpoint evaluated on the test fold once** (supersedes interim 4.1's 40 epochs /
patience 8); seed 42. AST: lr 5e-5 cosine, batch 16, 5 epochs per fold, same selection rule, precision
chosen by an automatic probe (D-27/D-28).
Metrics: accuracy, macro-F1, confusion matrix.

## 6. Efficiency, compression and benchmarking (interim 4.6)

* **Params / size:** trainable params; saved file size (fp32, int8, sparse reported separately).
* **Latency:** batch 1, one 4 s clip, CPU only, 20 warm-up + 200 timed runs, median and
  p95 at 1 and 4 threads, all models on one machine, feature extraction timed separately.
* **Quantization:** post-training 8-bit; dynamic for Linear/GRU (CRNN, AST); static with
  100 train-fold calibration clips for conv models; evaluated on all folds, no retraining.
* **Pruning:** global L1 unstructured at 30/50/70 % sparsity, 3 fine-tune epochs, folds
  1/5/9; report non-zero params and gzip size; unstructured sparsity does not speed up CPU.
* On-device mobile measurement: out of scope (limitation).
* Implementation of this section (routes, fine-tune rule, harness rules, order of commands): D-31..D-35 and `docs/RUNBOOK.md`.

## 7. Deliverables and deadlines

| Item | Date |
|---|---|
| Final presentation (10-12 min + Q&A, every member presents) | **26-29 Oct 2026** |
| Final report (Template Part C, 12+ references) + GitHub repo + signed contribution statement (C.3) | **Sat 31 Oct 2026, 11:59 PM, hard** |
| Turnitin: plagiarism <=15 %, AI content <=20 % | with the report |

Final report must contain: comparison table across all models on common metrics, error
analysis, efficiency metrics, discussion of *why* models differ, LLM-use disclosure with the
exact places. Marks: implementation + comparison 2.5, viva/individual 1.5 (+/-20 %),
report 1.0, presentation 1.0. Fabricated results = zero for the phase.

## 8. Environment (measured 2026-10-19)

* Apple M5, 10 cores (4 performance + 6 efficiency), 32 GB unified memory, macOS (Darwin 25.6),
  592 GB free disk, Python 3.13.5 (project `.venv`).
* torch 2.14.1 with **MPS available, no CUDA**; torchaudio 2.11, torchvision 0.29, timm 1.0.30,
  transformers 5.20.1, librosa 1.0.0, scikit-learn 1.9.1, pandas 3.0.6, matplotlib 3.11.2.
  (librosa, timm, transformers, pypdf, python-docx were not installed at the start; installed
  into `.venv`.)
* Matmul throughput probe (4096^2): MPS fp32 3.6, fp16 15.5, bf16 15.7 TFLOP/s; CPU fp32 2.1.
* AST (85.6 M params, 400 frames, random weights, training step incl. backward + AdamW):
  **~68 ms/clip with fp16 autocast, ~115 ms/clip fp32** on MPS, ~9 GB at batch 16.
* Quantization: only the `qnnpack` backend exists (engine must be set explicitly,
  otherwise `NoQEngine`); `torch.ao.quantization` is deprecated in favour of `torchao`.

## 9. Compute feasibility (estimates from the probes above, to be re-measured)

* **AST, one fold:** ~7,000 train clips x 5 epochs x 0.068 s = ~40 min + validation (~1 min).
  **10 folds ~ 7 h** (fp16) or ~12 h (fp32). Feasible, but it is a single-GPU serial job that
  must run unattended with `caffeinate` (system sleep is set to 1 minute on this Mac).
* AST pruning sweep as specified (3 sparsities x 3 folds x 3 epochs = 27 epoch-passes) ~ 3.5 h more.
* CNN / MobileNetV3 / CRNN: small; per-epoch time not yet measured (needs the pipeline).
  Expect minutes per fold, but 3 configs x 10 folds x up to 40 epochs is not free.
* CPU latency reference points (random weights, loaded machine, indicative only):
  AST fp32 ~99 ms (1 thread) / ~73 ms (4 threads); MobileNetV3-Small ~5 ms (1 thread).
* Because MPS is the only accelerator, training jobs cannot run in parallel without slowing
  each other, and **latency benchmarks must run on an otherwise idle machine, after training**.

## 10. Ambiguities and risks

**Schedule**
1. Presentations start **26 Oct**, five days before the report deadline. Real results must
   exist by ~24 Oct for slides; the plan below treats 24 Oct, not 31 Oct, as the compute deadline.
2. The synopsis work plan has "compression + benchmarking" due 20 Oct, which is already
   unattainable; everything from preprocessing onward is still to be built.

**Team / integrity**
3. The synopsis and interim assign one model per member (three members), with a signed task
   matrix and a per-member viva. If the work is done by one person, the Part C.3 contribution
   statement and commit history must say so truthfully; they are cross-checked against GitHub.
4. The submitted interim contains fabricated results (4.5). Not our decision here, but
   the final report must not echo them, and the user may want to decide whether to tell the faculty.

**AST**
5. The public HF checkpoint (`MIT/ast-finetuned-audioset-10-10-0.4593`) is built for **1024**
   frames. Using 400 frames needs its positional embeddings cropped or interpolated (as the
   authors' code does) because the HF class will not do it silently. This is a real
   implementation task, and a source of subtle bugs.
6. The checkpoint is AudioSet-finetuned. AudioSet has dog bark, siren, jackhammer, etc., so
   AST gets an in-domain advantage that the from-scratch CNNs do not. The pretrained
   MobileNetV3 control only covers ImageNet pretraining. State this; optionally add a
   PANNs-style audio-pretrained CNN if time allows (probably not).
7. **Fallback plan:** run folds in the order 1, 5, 9, 2, 3, ... so the fallback subset
   (1, 5, 9) exists first. Trigger: if measured per-fold wall time x 10 would not finish by
   24 Oct, stop after the folds completed and label the AST rows "n folds" in every table.

**Compression / benchmarking**
8. **Dynamic int8 AST on this Mac measured ~7x slower than fp32** (≈700 ms vs ≈98 ms,
   random weights, qnnpack). Quantization may therefore *shrink* models but *not* speed them
   up with PyTorch on Apple Silicon, so "quantized latency" would be misleading. Options: report
   the slowdown honestly as a finding, and/or use ONNX Runtime int8 for latency (a new dependency).
9. Static quantization of timm MobileNetV3 (hard-swish, squeeze-excite) is not eager-mode
   ready: needs Quant/DeQuantStub and FloatFunctional rewiring, or a different toolchain
   (torchao PT2E / ONNX Runtime). Significant effort for a secondary result.
10. Latency is measured on an Apple CPU (4P+6E cores), not an edge device. "4 threads" on a
    heterogeneous chip is noisy (a probe showed MobileNetV3-Small *slower* at 4 threads than
    1, while the machine was busy downloading). Needs an idle machine, many repeats, and an
    explicit statement of the limitation.
11. Unstructured pruning gives no size reduction in a dense file; the gzip-size reporting
    in the spec is the right call, but the sweep (every model x 3 sparsities x 3 folds x 3
    epochs) is heavy mostly because of AST. Candidate cut: prune AST only at 2 levels.

**Interim design: things that look wrong or inconsistent**
12. Citation numbers in the interim text do not match its own reference list (e.g. "paper [5]"
    is called AST in 4.4 but [5] is the CRNN paper; the CNN "[2] and [3]" order is swapped).
    Fix when writing the final reference list.
13. Class counts and "most clips are exactly 4 s": checked, both true (see section 12).
14. Mixed precision on MPS: fp16 autocast works for the AST probe; whether it is stable
    (no GradScaler on MPS) and supports BiGRU must be checked on the real models. bf16 is the
    safer fallback.
15. Resampling to 16 kHz drops content above 8 kHz and upsamples the low-rate clips; the
    interim justifies it for AST, but it is a limitation to state.
16. "Mean +/- std over folds" is fine, but macro-F1 should be defined as per-fold macro-F1
    averaged, with pooled-confusion-matrix F1 as a secondary number.
17. Literature table figures (e.g. 73 %, 79 % on UrbanSound8K, AST 95.6 % ESC-50, Deep
    Compression 35x/49x) are unverified here; check each against the paper before citing.

## 11. Schedule (owner-approved 2026-10-19; compute deadline is 24 Oct)

Projections use measured s/epoch (CNN 14.3, MobileNetV3-S 9.6, CRNN 14.4; AST ~480 from a probe) and the
fixed protocol-v2 epoch counts, so they are estimates, not bounds. Queue launched ~21:45 IST on 19 Oct.

| When (IST) | Work |
|---|---|
| 19 Oct, ~21:45 -> ~01:45 (20 Oct) | CNN family, 10 folds each, ~4.0 h: CNN 73 min, MobileNetV3-S scratch 48, MobileNetV3-S ImageNet 48, CRNN 72 (automatic) |
| ~01:45 | AST precision probe, ~4 min (automatic) |
| ~01:50 -> ~08:40 (20 Oct) | AST 10 folds in order 1,5,9,2,3,4,6,7,8,10, ~6.7 h (automatic, same queue) |
| 20 Oct, after the queue is idle | **Needs owner's go-ahead:** quantization (PyTorch and ONNX Runtime), pruning (AST 30/70 %, others 30/50/70 %), then the latency/size benchmark on an idle machine |
| 21-23 Oct | Compression runs, buffer for failures/re-runs |
| 24 Oct | **Compute deadline.** Comparison tables, confusion matrices, error-analysis figures (auto-generated) |
| 25 Oct | Slides |
| 26-29 Oct | Presentation |
| 30-31 Oct | Report (owner writes the prose), Turnitin, repo clean-up, submit by 31 Oct 11:59 PM |

AST fallback trigger: if the measured per-fold wall time x 10 cannot finish by 24 Oct, stop after
the completed folds and label the AST rows "n folds" in every table.

## 12. EDA findings (real data, `results/eda/`)

Verified: 8,732 WAVs, 10 folds, 10 classes, every metadata row has its file, MD5 of the archive
matches Zenodo. Folds hold 806-990 clips.

* **Interim class counts are correct:** gun_shot 374, car_horn 429, siren 929, all other
  classes 1,000 (sum 8,732).
* **"Most clips are exactly 4 s": true, 7,333 (84 %)**, but 1,399 are shorter (457 under 1 s) and
  8 are marginally longer (max 4.037 s). The short ones are concentrated in the minority
  classes: gun_shot 358 of 374, car_horn 226/429, dog_bark 325/1000. So zero-padding is not
  class-neutral, and a model can partly learn "how much silence" (see risk 18).
* Sample rates are very mixed: 44.1 kHz 5,370; 48 kHz 2,502; 96 kHz 610; 192 kHz 17; and 8-32 kHz
  for 233 clips. 7,993 clips are stereo, 739 mono, so downmix to mono is needed. Bit depths: PCM16,
  PCM24, float32, U8, ADPCM (all readable by soundfile).
* No all-zero clips; one near-silent siren clip (`159747-8-0-21.wav`, peak 0.00093).
* **Source leakage:** 1,297 source recordings; 5 of them (fsIDs 77751, 106905, 132016,
  176638, 180937) have clips in more than one fold, always as *different classes* cut from the
  same Freesound file. Folds are otherwise source-disjoint. It is the dataset's official split,
  so it is kept as-is and noted as a minor limitation, not fixed.

18. **Zero padding vs log-mel floor (new risk):** padded regions become the minimum of the dB
    scale (-100 dB with `amin=1e-10`), far below real content. The padding length is correlated with
    class (above), and it dominates per-round mean/std normalisation. Decide the floor/`top_db`
    clamp before building features; consider reporting accuracy separately for short vs full clips.
