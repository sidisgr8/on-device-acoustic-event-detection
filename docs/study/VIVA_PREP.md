# Viva preparation (study material, not report text)

Short, correct answers grounded in what this repo actually does. Numbers come from `METHODS_FACTSHEET.md` (read that first).
**Confidence tags:** none = I am confident and it is checkable in the repo; **[SOURCE]** = backed by a paper I read (see `REFERENCES_VERIFIED.md`);
**[NOT SURE]** = my explanation or recollection, not verified here, so say it as a hypothesis or check before the viva.
Learn the *reasoning*, not the wording. Examiners will follow up with "why" and "what would change".

---

## A. Shared pipeline and evaluation

**Why log-mel spectrograms as input?**
A waveform is a long 1-D signal (64,000 samples here) whose class information is in *which frequencies are active and how that evolves in time*. The STFT shows that as a 2-D image; the mel scale spaces
frequency bands like human hearing (fine resolution at low frequencies, coarse at high), which compresses 513 linear bins to 128 informative bands; the log (dB) compresses the huge dynamic range of sound
power so that quiet and loud parts are both visible. Result: a 128 x 201 "image" that CNNs/transformers can process and that every model here shares (AST uses a very similar Kaldi fbank).

**Why 16 kHz?**
It keeps the speech/environmental-sound band up to 8 kHz (Nyquist) and is what the pretrained AST checkpoint expects, so one resampled copy serves all models. Cost: content above 8 kHz is lost (e.g. high harmonics, hiss), and the 58 clips recorded at 11.025 kHz or below are upsampled
and carry no information above their original Nyquist (their empty high bands are one reason the dB floor matters). The loss is a limitation, not a bug.

**Why n_fft 1024, hop 320, 128 mels, fmin 20, fmax 8000?**
These come from the interim design and were not tuned. Meaning: window 1024/16000 = 64 ms (frequency resolution 15.6 Hz; long enough to resolve low-frequency tones like engine hum, short enough to follow changes), hop 320 = 20 ms (frame rate 50 Hz, 68.75 % overlap between consecutive windows),
128 bands (finer than the common 64), fmin 20 Hz (below this there is no useful audible content), fmax 8000 = Nyquist. `center=True` pads so frame t is centred at t x hop: 1 + 64000/320 = 201 frames.
If asked "why not 64 mels": design choice from the spec; more bands cost more compute (the first conv layers dominate the CNN's 361 M MACs) and I have no ablation, so say so.

**Why power to dB (log)?**
Perceived loudness is roughly logarithmic and raw spectrogram power spans many orders of magnitude; the log makes the distribution closer to well-behaved for gradient-based learning and lets multiplicative effects (gain, channel) become additive shifts. We use 10 log10(power) with a tiny floor (1e-10) to avoid log(0).

**What is the -80 dB floor and why?**
Zero padding of short clips is digital silence = exactly -100 dB, far below real content (median -24 dB), so it is an extreme outlier that also distorts normalisation. We clamp values below -80 dB (D-15). The value comes from a rule fixed in advance: the highest multiple of 10 dB that clamps at most 1 % of real bins
in the round-1 training folds (0.58 % at -80). Decided from audio statistics only, no model results. **Honest limit:** padding moves from 4.4 to 3.3 sigma below the mean; it is still detectable (see below), because you cannot hide padding without changing the input length.
Alternative rejected: per-clip `top_db` clamp, which would erase absolute loudness (informative for gun shots, sirens) and make the padding level depend on the clip.

**How is normalisation done and why that way?**
One scalar mean and std per round, computed on the training folds only and only on real (non-padded) frames, then applied to train/val/test (D-16). Training-fold-only prevents test information leaking into the preprocessing. Scalar (not per mel band) keeps the relative spectral shape intact and matches common practice.
Excluding padded frames keeps the statistics independent of how much padding the training clips contain, which is class-correlated. AST instead uses the checkpoint's fixed constants (mean -4.27, std 4.57, divided by 2x std) because the pretrained weights expect that input scale.

**What is the fold rule and why does it prevent leakage?**
Round k: test = fold k, validation = fold (k mod 10)+1, train = the other 8; the dataset's predefined folds are never re-split. UrbanSound8K clips are slices of longer Freesound recordings; slices of the same recording share background noise, microphone and
sometimes the same sound source. A random clip-level split would put near-duplicates in both train and test and inflate accuracy. The predefined folds keep slices of one recording together **[SOURCE: could not verify the original paper's wording, see REFERENCES_VERIFIED]**; our EDA confirms it for
1,292 of 1,297 recordings. The 5 exceptions (e.g. fsID 77751 appears as drilling in fold 2 and jackhammer in fold 7) are a small residual leak we report. Unit tests check train/val/test never overlap for all 10 rounds.

**Why a separate validation fold, selection on validation, and the test fold used once?**
Choosing the best epoch on the test fold would make the reported number optimistic (selection bias). So selection uses the validation fold; the test fold is touched once per round by the best-validation checkpoint (the result file is created in exclusive mode so a second evaluation is impossible). Rule D-20: no hyper-parameter or architecture decision may use test results.

**Why report both accuracy and macro-F1?**
Classes are imbalanced (gun_shot 374, car_horn 429, siren 929 vs 1,000 for the rest). Accuracy weights every clip equally, so a model can look good by doing well on the big classes. Macro-F1 averages per-class F1 (harmonic mean of precision and recall) with equal weight per class, so poor minority-class
performance shows up. We compute macro-F1 per test fold and average over folds; the confusion matrix is summed over folds.

**What does "mean +/- std over folds" tell you, and what does it not?**
It shows variation across different test sets (data variability) with one training run each. It does not capture variation from random initialisation or data order because only seed 42 is used. With 10 folds the std is itself noisy. Do not present it as a confidence interval over seeds. Folds differ in size (806-990 clips) and are averaged unweighted.

**What is SpecAugment and why use it?**
Randomly masking a few frequency bands and time spans of the input spectrogram [SOURCE: Park et al. 2019, for ASR]. It is a cheap regulariser: the model cannot rely on one narrow band or one instant, so it must use redundant cues, which reduces overfitting on a small dataset (about 7,000 training clips per round). We use only the masking part: 2 frequency masks (<= 16 bins) and 2 time masks (<= 20 frames),
training only, masked values set to 0 (= the mean after normalisation). No time warping, no waveform augmentation (the interim kept those as optional ablations). Equalised in seconds: the time-mask limit is 400 ms for every model, i.e. 20 frames for the 20 ms-hop log-mel models and 40 frames for AST (10 ms shift) (D-36, decided before any AST result existed, so it cannot be a result-driven tweak).

**Why label smoothing 0.1?**
Instead of a one-hot target the loss uses (1 - 0.1) on the true class plus 0.1 spread uniformly (PyTorch adds 0.1/10 to every class). It stops the network driving the logits to extreme values, acts as a regulariser and usually improves calibration [SOURCE: Szegedy et al. 2016 introduce it; Muller et al. 2019 analyse it]. Side effect: the training loss can never reach 0
(the floor is about 0.5 for 10 classes), so do not read the training loss as "not converged".

**Why AdamW, weight decay 0.01, warm-up and cosine decay?**
AdamW decouples weight decay from the adaptive gradient step, which regularises more correctly than L2-in-the-loss with Adam [SOURCE: Loshchilov & Hutter]. Warm-up (2 epochs) avoids large, noisy early updates when BatchNorm statistics and Adam's moment estimates are not yet reliable. Cosine decay lowers the learning rate smoothly to 0, which tends to settle the model at the end of training.
Here the cosine is defined over exactly the fixed number of epochs. Note: decay is applied to all parameters including biases and norms (a simplification).

**Why did the protocol change from "40 epochs with early stopping" to "30 fixed epochs"? (be ready to be honest)**
The first CNN fold stopped at epoch 14 of a 40-epoch cosine schedule, so the learning rate had barely decayed, and validation accuracy swung by up to ~10 points between epochs; patience-8 early stopping fired on noise. The owner changed all CNN-family models to 30 fixed epochs with a schedule that completes and keeps the best-validation checkpoint,
matching AST's design. The fold-1 test accuracy of the aborted run had been seen before the decision; the justification is the schedule mechanics and the validation curves. The aborted outputs are kept, labelled and unused (D-26).

**Why is accuracy alone misleading for edge deployment? What do you report instead?**
On a device the constraints are memory, compute, latency and energy. Two models with similar accuracy can differ by orders of magnitude: AST has 85.6 M parameters, 343 MB in fp32 and about 44 G MACs per clip; the CNN baseline has 0.39 M parameters and 1.6 MB; MobileNetV3-Small 1.53 M parameters.
We report parameters, file size, multiply-accumulates (computed), and measured CPU latency (median and p95, batch 1, 1 and 4 threads, feature extraction separate), plus the same after quantization and pruning. **Careful:** parameters are not compute: the CNN baseline has 4x fewer parameters than MobileNetV3-Small but about 12x more MACs (361 M vs 28.6 M).

**What does the padded-vs-full accuracy table show and why is it there?**
Short clips are zero-padded to 4 s. Padding length is class-correlated (96 % of gun_shot clips are padded, about 0 % of air_conditioner, street_music), and non-acoustic metadata (sample rate group, channels, sample format, duration) alone predicts the class at 19.9 % vs 10 % chance. So a model can gain a little from "how much silence". Reporting accuracy separately for padded and full clips lets you check
whether performance depends on padding.

---

## B. CNN baseline

**Describe the architecture and the shapes.**
4 x [Conv 3x3 (padding 1, no bias) - BatchNorm - ReLU - MaxPool 2x2], channels 32/64/128/256: (1,128,201) to (32,64,100) to (64,32,50) to (128,16,25) to (256,8,12); average over all positions to a 256-vector; Dropout 0.3; Linear 256 to 10. 390,890 parameters.

**Why BatchNorm?**
It normalises each channel's activations over the batch, which stabilises and speeds up training, lets you use a larger learning rate, and has a mild regularising effect (batch statistics add noise). [SOURCE: Ioffe & Szegedy, not read here.] Careful: "reduces internal covariate shift" is the original explanation but is debated; safer to say it smooths optimisation. At test time it uses running statistics (so batch size 1 inference is fine).
Conv layers have no bias because BN's shift makes it redundant.

**Why ReLU?** Cheap, non-saturating (no vanishing gradient for positive inputs), sparse activations; it also quantises well. (MobileNetV3 uses hard-swish instead.)

**Why 3x3 kernels and max pooling?**
Stacked small kernels give a large receptive field with few parameters: 3x3 is the smallest kernel that captures neighbourhood in both axes. Max pooling halves resolution (cheaper later layers) and gives small shift invariance in time and frequency. After four blocks each output cell sees a **46 x 46** patch of the input (46 mel bins x 46 frames = 0.92 s)
by the recursive receptive-field formula (r = 3, 4, 8, 10, 18, 22, 38, 46).

**Why global average pooling instead of flattening?**
Flattening 256 x 8 x 12 would need a 24,576 x 10 linear layer (about 246 k parameters, equal to 63 % of the current total) and would tie the classifier to the exact input length. Averaging over frequency and time makes the head tiny (2,570 parameters), is robust to where an event occurs in time, and acts as a regulariser. Cost: it discards precise position information.

**Why dropout 0.3, and where?** Only before the classifier, to regularise the final linear layer on a small dataset; 0.3 is the spec's value, untuned.

**Few parameters, but is it cheap?**
No: 361 M MACs per clip. Channels double while resolution halves, so every block from the second on costs about 118 M MACs (7 M for the first). MobileNetV3-Small needs 28.6 M. Parameter count is a poor proxy for latency; that is why latency is measured.

---

## C. MobileNetV3-Small (scratch and ImageNet)

**What makes MobileNetV3 efficient?**
Depthwise-separable convolutions: a K x K depthwise conv (one filter per channel) followed by a 1x1 pointwise conv replaces a full K x K conv. The multiply-accumulate cost ratio is about 1/C_out + 1/K^2 (about 1/9 for 3x3 with many channels). Plus inverted residual blocks (expand with 1x1, depthwise 3x3, project with a linear 1x1 bottleneck), squeeze-and-excite (a tiny channel-attention gate),
hard-swish activations (a cheap piecewise approximation of swish), and architecture found by hardware-aware search (NAS + NetAdapt) [SOURCE: Howard et al. 2019; MobileNetV2 for inverted residuals].

**Why the Small variant?** The spec's "edge" operating point: 1,527,818 parameters and 28.6 M MACs. MobileNetV3-Small is documented as 6.6 % more accurate than a MobileNetV2 of comparable latency on ImageNet [SOURCE, final version].

**How was the single-channel input handled?**
The first conv expects 3 RGB channels; for a 1-channel spectrogram timm sums the three pretrained filters into one (exactly equivalent to feeding the same image in all three channels). For the scratch model the stem is simply 1-channel. We verified the equivalence numerically in a unit test.

**Why does the ImageNet-pretrained run exist (the "pretraining control")?**
AST is pretrained (ImageNet then AudioSet) and the other models are not, so a gap between them mixes architecture with pretraining. Running MobileNetV3-Small both from scratch and from ImageNet weights measures how much pretraining alone helps a small CNN on spectrograms.
**Honest caveats:** (1) ImageNet is images, not audio, and our inputs are standardised log-mel, not ImageNet-normalised RGB; (2) the pretrained run also uses lr 3e-4 instead of 1e-3, so two things change at once; (3) it does not equalise AST's *audio-domain* AudioSet pretraining.

**Why would ImageNet weights help on spectrograms at all?** [NOT SURE, general belief] Early layers learn edge/texture/oriented-pattern detectors that are also useful on time-frequency images, and a good initialisation speeds convergence on a small dataset. Whether it helps here is an empirical question answered by the control run.

**Fewer MACs than the CNN. Will it be faster?**
Not necessarily [NOT SURE until measured]: depthwise convolutions have low arithmetic intensity and are often memory-bound, and squeeze-excite/hard-swish add small ops. MAC count is a proxy; the latency table decides.

---

## D. CRNN

**Why a convolutional front end then a recurrent layer?**
Convolutions extract local time-frequency patterns (a siren sweep, a drill's harmonics); the recurrent layer then models how those features evolve over the clip. Here the convs reduce (128 x 201) to (128 channels x 16 freq x 50 time steps); each time step is flattened to a 2,048-vector, projected to 128, and fed to the GRU [SOURCE: Cakir et al. 2017 for the idea, in polyphonic event detection].

**Why a GRU and not an LSTM?**
A GRU has two gates (update, reset) and no separate cell state, so about 25 % fewer parameters than an LSTM of the same size (3 weight blocks instead of 4 per direction); it often performs similarly. The CRNN paper chose GRU because "preliminary experiments using LSTM yielded similar results and GRU units have a smaller number of parameters" [SOURCE].
We did **not** run an LSTM ablation (it was optional), so do not claim to have compared them.

**Why bidirectional?** For whole-clip classification the entire 4 s is available, so the backward pass can use future context. Consequence: the model is not causal, so it cannot run on a live stream frame by frame without buffering a window (a limitation for real-time use; the chosen task is clip classification).

**Why concatenate mean and max pooling over time?**
Mean pooling summarises sustained evidence (engine idling, air conditioner); max pooling captures the strongest brief evidence (a dog bark, a gun shot). Concatenating gives the classifier both (512 = 2 x 256). It also removes dependence on clip length and avoids relying on the last hidden state alone.

**Check the shapes.** (B,1,128,201) -> convs (B,128,16,50): pooling (2,2),(2,2),(2,1) halves frequency three times (128 to 16) and time twice (201 to 50) -> per-step vector 128 x 16 = 2,048 -> Linear to 128 -> BiGRU (B,50,256) -> [mean;max] (B,512) -> Linear 10. Parameters: GRU 198,144; projection 262,272; total 558,442.

**What problem do gates solve?** Plain RNNs multiply gradients through many steps and they vanish or explode; gates give additive paths so information and gradients can be carried over many steps. With only 50 steps this matters less than for long sequences.

---

## E. Audio Spectrogram Transformer (AST)

**Explain how AST turns a spectrogram into tokens.**
The 400 x 128 (time x mel) fbank is cut into overlapping 16 x 16 patches with stride 10 in both axes (overlap 6) by one strided convolution (1 -> 768 channels): 12 frequency x 39 time = 468 patch tokens. Two learnable tokens ([CLS] and a distillation token) are prepended: 470 tokens of dimension 768. A learnable position embedding (1 x 470 x 768) is added, then 12 transformer encoder layers
(multi-head self-attention, 12 heads, MLP 3072, LayerNorm, residuals). The two special-token outputs are averaged, normalised and fed to a linear layer (here 768 to 10) [SOURCE: Gong et al. 2021].

**Why are position embeddings needed?** Self-attention is permutation-invariant: without them the model would see an unordered bag of patches. The learned embedding tells each token where in frequency and time its patch came from.

**Why did you crop the position embeddings, and how exactly?**
The checkpoint was trained on 1024 frames (a 12 x 101 patch grid, embedding tensor 1 x 1214 x 768). Using 400 frames needs a 12 x 39 grid (1 x 470 x 768). Re-initialising would throw away the pretrained position information (HuggingFace does that silently), so we cut it as the AST authors' code does: keep the 2 special-token embeddings, reshape the 1212 patch embeddings
to (12, 101, 768), keep all 12 frequency rows and take time columns [31, 70) (a centred crop, index 50 - 39//2 = 31), flatten back. This relies on the token order being frequency-major; we proved it with an impulse test (an impulse at time frame 5, mel bin 125 changes exactly token 429 = 11 x 39 + 0).
Unit tests also check the cropped values equal the checkpoint slice and that the other 200 pretrained tensors load bit-identically.

**Why a centred crop and not the first 39 columns?** The authors' code does a centred crop (I read their source). **[NOT SURE why]** - plausible reasons: the central time positions were trained on every 10 s clip and are least affected by edge effects; I did not test the alternative and `pos_crop: first` exists in the config but was not run. Do not claim one is better.
(The AST *paper* describes "cut and bi-linear interpolate" for ImageNet ViT to AST, which is a different adaptation step [SOURCE].)

**Why 400 frames instead of the checkpoint's 1024?** Clips are 4 s = 400 frames at 10 ms; padding to 1024 would make 1,212 patch tokens, mostly padding, and attention cost grows with the square of the token count: 470 vs 1,214 tokens means about 2.6x fewer tokens (about 6.7x cheaper attention matmuls, 2.6x cheaper linear layers).

**What does fine-tuning from AudioSet/ImageNet change compared with training from scratch?**
The weights start from a model that already learnt general audio patterns from AudioSet (roughly 2 million 10-second clips **[NOT SURE of the exact count; check the AudioSet paper]**) (and before that, visual features from ImageNet via DeiT initialisation). With only about 7,000 training clips per round, a 85.6 M-parameter transformer would overfit badly from scratch; here only the 10-way head is new and everything is fine-tuned at a small learning rate (5e-5) for 5 epochs.
Small lr protects pretrained features from being destroyed by large early gradients. "Lineage": ImageNet (DeiT) -> AudioSet -> UrbanSound8K [SOURCE: AST paper, Table 3 uses DeiT with distillation].

**What is the caveat with AudioSet pretraining?**
AST has audio-domain pretraining with classes similar to ours (dog, siren, jackhammer, engine...), so it has an advantage the from-scratch CNNs lack; the ImageNet MobileNet control does not equalise that. I did not check whether UrbanSound8K source recordings (Freesound) overlap with AudioSet (YouTube) content.

**What is the cost of self-attention?** O(T^2 d) for T tokens: here 470 tokens, so the attention score and value matmuls add about 4.1 G MACs on top of about 40 G MACs in linear/conv layers: roughly 44 G MACs per clip versus 0.03-0.36 G for the CNN-family models. That is the central accuracy-efficiency tension of the project.

**Why fp16/bf16 for AST and how was it chosen?** A transformer this size is memory- and compute-heavy; reduced precision is faster on this GPU (probe: roughly 68 ms/clip vs 115 ms in fp16 vs fp32). To avoid assuming it is safe, the queue runs a probe: 30 optimiser steps on fold-1 *training* clips in fp32/fp16/bf16, choosing the fastest mode whose loss is finite and within 0.10 of fp32 (rule fixed in advance, D-27; result in D-28).

---

## F. Compression and efficiency

**Dynamic vs static quantization: what is the difference?**
Both store weights as 8-bit integers. *Static* quantization also quantizes **activations** with scale/zero-point fixed beforehand from a calibration run (here 100 training clips), so inference uses integer kernels end to end. *Dynamic* quantization stores int8 weights but computes each layer's activation scale **on the fly** from the actual input at run time, so no calibration data is needed; it suits Linear/recurrent layers where activations vary
and weight loading dominates. We use static for the conv models (CNN, MobileNetV3) and dynamic for Linear/GRU layers (CRNN, AST). [Reading suggestion: Krishnamoorthi's whitepaper, not read here.]

**Post-training quantization vs quantization-aware training?**
PTQ converts a trained model without retraining (fast, may lose accuracy); QAT simulates quantization during training so the model adapts (usually better accuracy, needs training). Jacob et al. 2018 is QAT [SOURCE]. We do PTQ only (spec), so say that your numbers are PTQ and may be conservative.

**What is calibration?** Running representative data through the float model to record activation ranges (min/max or histogram) so each activation tensor can be mapped to 8 bits. We use 100 clips from the training folds; using test clips would leak.

**Why also ONNX Runtime?** PyTorch's int8 path on this machine (qnnpack on Apple silicon) may not be representative of a deployment runtime. ONNX Runtime is a common deployment engine, so it gives a "deployment-style" int8 number. We always report size reduction and latency **separately** (D-09) because they can move in opposite directions.

**Why can PyTorch dynamic int8 be slower than fp32 here?** The measured fact: in a random-weight probe on this Mac, AST fp32 took about 98 ms and dynamic int8 about 702 ms per clip (1 thread). **[NOT SURE of the cause]**: my hypothesis is that PyTorch's qnnpack dynamic Linear path on arm64 macOS is far less optimised for the large matrix multiplications of a transformer than the fp32 path (which can use Apple's highly tuned matrix routines),
and dynamic quantization adds per-call activation quantization overhead. I did not profile it. The point to make: quantization reduces *size* reliably but reduces *latency* only if the runtime has fast int8 kernels for that hardware.

**What does ONNX Runtime do with the CRNN's GRU?** ORT has no dynamic int8 for GRU, so the GRU stays float32 in the ORT int8 model (recorded in the result file); only the Linear/MatMul layers are quantized. PyTorch dynamic quantization does quantize the GRU.

**Expected size reduction from int8?** About 4x for the quantized weights (4 bytes to 1 byte) minus overhead for scales/zero-points and anything left in float (BatchNorm, GRU in ORT, QDQ nodes). Models with other float parts show less than 4x. In the dry runs (CNN on its real validation checkpoint, the others on random weights; sizes do not depend on the weights) the size ratios were about 3.9x (CNN, FX static), 3.5x (MobileNetV3-S, FX static), 3.9x (AST, PyTorch dynamic), 2.6x (CRNN, PyTorch dynamic) and 1.9x (CRNN, ORT, where the GRU stays float32).

**What does magnitude pruning do?** Sets the weights with the smallest absolute value to zero (here globally: one threshold across all conv/linear/GRU weights, so layers end up with different sparsity), then fine-tunes 3 epochs with the mask held fixed so the network can recover. Rationale: small weights contribute little to the output.

**Why doesn't unstructured pruning speed up CPU inference?** The zeros are scattered, and standard dense kernels still multiply them; speedups need sparse kernels/formats or hardware support, or *structured* pruning (removing whole channels/heads, which changes the architecture). What does shrink: the count of non-zero parameters and the *compressed* (gzip) file size, because zeros compress well. We state this in the results.

**Why global rather than per-layer pruning?** One threshold lets the data decide where redundancy is (a unit test shows a layer with large weights keeps most of them). Per-layer pruning would force the same sparsity everywhere, often hurting sensitive layers (first conv, classifier) more.

**Deep Compression reports 35x-49x. Why will you not see that?** Those factors come from pruning *plus* 5-bit trained quantization *plus* Huffman coding together [SOURCE]. We prune only (dense storage) and quantize separately to 8 bits, so expect smaller factors (gzip size roughly tracks the non-zero fraction).

**How is latency measured and what are the caveats?**
Batch 1, one 4 s clip, CPU, 20 warm-up and 200 timed runs, median and p95, at 1 and 4 threads, feature extraction timed separately from the forward pass, on an otherwise idle machine (the tool refuses to run otherwise). Caveats: Apple M5 has 4 performance + 6 efficiency cores and macOS gives no core pinning, so "4 threads" is not tied to the fast cores and is noisier; tiny models can be slower with more threads;
Apple silicon is not an edge CPU, so numbers compare models rather than predict a phone. Not yet run when this was written.

---

## G. Honest limitations (be the first to say them)

1. **Small residual source leakage** across folds (5 of 1,297 recordings).
2. **Padding/clip length is a weak shortcut** (metadata alone gives 19.9 % vs 10 % chance); mitigated by reporting padded vs full accuracy, not removed.
3. **Different front-ends and normalisations** for AST vs the rest; different lr/batch size. (SpecAugment durations are equalised at 400 ms, D-36.)
4. **AST has audio pretraining** the other models lack; the ImageNet control is confounded (lr, domain).
5. **One seed**; std over folds is not a confidence interval over initialisations; unweighted fold means.
6. **Validation fold is small** (+/- ~1.5 points sampling noise), so best-epoch selection is noisy.
7. **Protocol v2 was adopted after seeing one fold's test number** (disclosed in D-26); the justification was the validation curves and schedule mechanics.
8. **Latency on one non-edge machine**, noisy multi-thread numbers; PyTorch int8 not representative here.
9. **Resampling to 16 kHz** discards >8 kHz content; AudioSet/ImageNet overlap not checked.
10. **No hyperparameter tuning**: values come from the spec; results are for these settings, not the best achievable.

## H. To prepare after the results exist (do not answer from guesses)

* Which classes are confused and why (look at `results/figures/confusion_*.png`, `top_confusions.png`; think about acoustic similarity, e.g. steady broadband noise classes, short impulsive classes).
* Where AST gains over the CNN family and by how much per class; whether the gain survives on the padded subset.
* Pretrained vs scratch MobileNetV3: how large is the pretraining effect, and what does that say about the AST gap?
* Quantization: accuracy drop per variant, size ratio, and the measured latency of PyTorch vs ONNX int8 for each model.
* Pruning: sparsity at which accuracy drops, and why latency is unchanged.
* The accuracy-latency-size trade-off: which model you would deploy on a phone and why, given the numbers.
