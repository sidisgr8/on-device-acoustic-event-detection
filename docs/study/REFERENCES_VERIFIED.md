# Literature verification (study material, not report text)

Purpose: check every claim in the old interim literature table against the real source. The interim's results claims were
unverified and LLM-drafted; this file records what the papers actually say.

## How each item was checked (and what to trust)

* **Metadata** (title, authors, dates, venue comment): arXiv's structured API (`export.arxiv.org`), and **Crossref** (`api.crossref.org`) for venue, volume, pages, year, DOI.
* **Numbers and method details**: I downloaded the PDFs (arXiv; the ICCV open-access PDF for MobileNetV3; Piczak's author-site PDF), extracted the text locally and read the passages.
  Quotes below are short excerpts from that text; section/table numbers are as printed in those PDFs.
* **Not trusted:** an automatic web summariser I tried first gave MobileNetV3 numbers (15 % / 4.6 %) that turned out to come from the *version 1* abstract (see paper 6), and it could not read PDFs.
  Nothing below relies on it. Where a paper could not be read, the verdict says so.
* The old table's page numbers and venues for all 10 papers matched Crossref (details per paper).

## 1. Summary: old interim claim vs what the source says

| # | Paper | Old claim (interim table) | What the paper actually says | Verdict |
|---|---|---|---|---|
| 1 | Salamon, Jacoby, Bello 2014 | "Establishes the 10-fold protocol that prevents leakage between clips from the same source recording" | Paper not readable (paywalled). The dataset README (shipped with the data) says only that files are "pre-sorted into ten folds ... to help in the reproduction of and comparison with" the paper's results | **COULD NOT VERIFY** the leakage wording (venue/pages verified; our own EDA supports it for 1,292 of 1,297 recordings) |
| 2 | Salamon & Bello 2017 | "About 79 % with augmentation, above the same CNN without augmentation and a dictionary-learning baseline"; "three conv layers with small kernels"; augmentations TS, PS, DRC, noise | 0.79 mean accuracy with augmentation; 0.73 without; SKM dictionary baseline 0.74 without augmentation, below SB-CNN with it; 3 conv layers with 5x5 filters; TS, 2x PS, DRC, background noise | **VERIFIED** (nuance: without augmentation SB-CNN is 0.73, not above SKM's 0.74) |
| 3 | Piczak 2015 | "About 73 % on UrbanSound8K (as reported in later comparisons)"; shallow CNN on log-mel segments with deltas, averaged over segments; beat MFCC baselines | Own best variant **73.1 %** (LP); 2 conv + 2 FC layers; log-mel 60 bands x 41 frames with deltas; probability voting over segments; "outperforms baseline implementations relying on" MFCC | **VERIFIED** (the 73.7 % printed next to it belongs to Salamon & Bello's system) |
| 4 | Çakır et al. 2017 | datasets: TUT-SED Synthetic 2016, TUT-SED 2009, CHiME-Home; "CRNN outperformed CNN-only and RNN-only baselines on the evaluated SED sets" | **Four** datasets (also TUT-SED 2016 development); CRNN "consistently outperforms CNNs, RNNs and the two baseline methods on all three datasets for the main metric" (CHiME-Home uses equal error rate) | **DIFFERS (minor):** one dataset omitted, claim holds for the main metric on three of the four |
| 5 | Gong et al. 2021 (AST) | "0.485 mAP (ensemble) on AudioSet, 95.6 % on ESC-50, 98.1 % on Speech Commands V2" | 0.485 = Ensemble-M (single model **0.459**); ESC-50 **95.6 +/- 0.4** is AST-P (ImageNet + AudioSet pretraining), 88.7 for ImageNet only; Speech Commands V2 **98.11** is AST-S (ImageNet only; AudioSet-pretrained is 97.88) | **VERIFIED** with qualifiers (which model/setting each number belongs to) |
| 6 | Howard et al. 2019 (MobileNetV3) | "Large is 3.2 % more accurate than MobileNetV2 with 20 % lower latency" | Final arXiv v5 and ICCV camera-ready: 3.2 % more accurate, **20 %** lower latency; Small **6.6 %** more accurate at comparable latency. arXiv **v1** said 15 % and 4.6 %/5 % | **VERIFIED** against the final version (numbers changed between versions: cite the ICCV/v5 text) |
| 7 | Jacob et al. 2018 | "8-bit integer quantization of weights and activations with a quantization-aware training scheme; improved accuracy-latency trade-off on mobile CPUs; ImageNet, COCO" | Matches: integer-arithmetic-only inference plus a co-designed training procedure; ImageNet classification and COCO detection; MobileNet latency-vs-accuracy on Qualcomm Snapdragon 835 LITTLE cores (Fig. 1.1), ARM NEON | **VERIFIED** (note: it is quantization-**aware training**; this project uses post-training quantization) |
| 8 | Kong et al. 2020 (PANNs) | "CNN14 ... 0.439 mAP on AudioSet at publication" | **0.439 is the Wavegram-Logmel-CNN**; **CNN14 is 0.431** (Table IV); best previous system 0.392; transfer datasets in the text are ESC-50, DCASE 2019 T1, DCASE 2018 T2, MSoS, GTZAN (not UrbanSound8K) | **DIFFERS:** the 0.439 is attributed to the wrong model |
| 9 | Park et al. 2019 (SpecAugment) | "Time warping, frequency and time masking ... LibriSpeech, Switchboard; state-of-the-art ASR results" | Matches: 6.8 % WER (test-other, no LM) / 5.8 % with shallow fusion on LibriSpeech 960h; 7.2/14.6 % (no LM) and 6.8/14.1 % (fusion) on Switchboard/CallHome; previous best hybrids 7.5 % and 8.3/17.3 % | **VERIFIED** (this project uses only the masking part, on spectrograms of a different task) |
| 10 | Han, Mao, Dally 2016 | "35x (AlexNet) and 49x (VGG-16) size reduction without accuracy loss; ImageNet, MNIST" | AlexNet 240 MB to 6.9 MB (35x), VGG-16 552 MB to 11.3 MB (49x); also LeNet-300-100 (40x) and LeNet-5 (39x) on MNIST; "without loss of accuracy" per Table 1 | **VERIFIED** (the factors need pruning + 5-bit trained quantization + Huffman coding together; pruning alone is 9x to 13x fewer connections) |

---

## 2. Details per paper

### [1] J. Salamon, C. Jacoby, J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research"
* **Venue (Crossref):** Proc. 22nd ACM International Conference on Multimedia (ACM-MM), 2014, pp. 1041-1044. DOI 10.1145/2647868.2655045. Authors confirmed also by Semantic Scholar (open-access status: closed).
* **Link:** https://doi.org/10.1145/2647868.2655045 (ACM returned HTTP 403; the author-hosted PDF host did not resolve; no readable copy was obtained).
* **What is verified from other primary material:** the dataset README (in `data/UrbanSound8K/UrbanSound8K_README.txt`): 8,732 excerpts of at most 4 s, 10 classes, drawn from field recordings uploaded to Freesound, pre-sorted into folds 1-10 "to help in the reproduction of and comparison with the automatic classification results reported in the article"; file name format `[fsID]-[classID]-[occurrenceID]-[sliceID]`; sampling rate, bit depth and channels are those of the original upload. The SB-CNN paper (2) describes the folds as "10 stratified folds".
* **Verdict:** bibliographic details VERIFIED. The specific claim about preventing source-recording leakage COULD NOT VERIFY from the paper. Our EDA: 1,297 source recordings, 5 of them have clips in more than one fold (always as different classes).

### [2] J. Salamon, J. P. Bello, "Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification"
* **Venue (Crossref):** IEEE Signal Processing Letters, vol. 24, no. 3, pp. 279-283, 2017. DOI 10.1109/LSP.2017.2657381. arXiv 1608.04363 (v2, 28 Nov 2016). Link: https://arxiv.org/abs/1608.04363
* **Method (Section II):** "3 convolutional layers interleaved with 2 pooling operations, followed by 2 fully connected (dense) layers". Input: 128x128 time-frequency patches (3 s) from a log-scaled mel-spectrogram with 128 bands over 0-22,050 Hz, 1024-sample window at 44.1 kHz, hop of the same duration. Layers: 24, 48, 48 filters of 5x5; (4,2) max-pooling after the first two; 64-unit dense layer. Augmentations (Section II-B): time stretching {0.81, 0.93, 1.07, 1.23}; pitch shifting PS1 {-2,-1,1,2} and PS2 {-3.5,-2.5,2.5,3.5} semitones; dynamic range compression (4 presets); background noise (4 scene types).
* **Dataset / protocol:** UrbanSound8K, 10-fold cross-validation; "1 of the 9 training folds in each split as a validation set for identifying the training epoch that yields the best model" (this is the same scheme as ours: 8 train / 1 validation / 1 test).
* **Key results (Section III, Fig. 1):** without augmentation "mean accuracy of 0.74, 0.73 and 0.73 for SKM, PiczakCNN and SB-CNN"; with all augmentations SB-CNN reaches "a mean accuracy of 0.79", significantly above SKM with augmentation (p = 0.0003, paired two-sided t-test).
* **Verdict:** VERIFIED.

### [3] K. J. Piczak, "Environmental Sound Classification with Convolutional Neural Networks"
* **Venue (Crossref):** 2015 IEEE 25th International Workshop on Machine Learning for Signal Processing (MLSP), pp. 1-6. DOI 10.1109/MLSP.2015.7324337. Link: http://karol.piczak.com/papers/Piczak2015-ESC-ConvNet.pdf (author site; PDF read).
* **Method (abstract and Section 3.2):** "2 convolutional layers with max-pooling and 2 fully connected layers" on "segmented spectrograms with deltas". Log-scaled mel-spectrograms (clips resampled to 22,050 Hz, window 1024, hop 512, 60 mel bands), split into 50 %-overlapping segments of 41 frames ("short") or longer segments ("long"), 60x41 plus deltas; clip prediction by majority vote or probability voting.
* **Datasets:** ESC-50, ESC-10, UrbanSound8K (10-fold CV with one training fold as validation).
* **Key results:** on UrbanSound8K the best own variant, long segments with probability voting (LP), reaches **73.1 %** ("LP - 73.1%, US - 73.7%", where US is the Salamon & Bello unsupervised system); ESC-50 baseline 44 % vs best network 64.5 %. The paper states that simple augmentation "proved to be unsatisfactory for the UrbanSound8K dataset", so no augmentation was used there.
* **Verdict:** VERIFIED.

### [4] Y. Gong, Y.-A. Chung, J. Glass, "AST: Audio Spectrogram Transformer"
* **Venue (Crossref):** Proc. Interspeech 2021, pp. 571-575. DOI 10.21437/Interspeech.2021-698. arXiv 2104.01778 (v3, 8 Jul 2021). Link: https://arxiv.org/abs/2104.01778
* **Method (Sections 2.1-2.2):** audio converted to 128-dimensional log Mel filterbank features "computed with a 25ms Hamming window every 10ms"; split into 16x16 patches "with an overlap of 6" in time and frequency, linearly projected, learnable positional embedding added, a [CLS] token; initialised from a DeiT (ImageNet) model; the two DeiT [CLS] tokens are averaged. **Positional embedding adaptation (Sec 2.2):** "cut and bi-linear interpolate" the ImageNet ViT/DeiT 2-D embedding to the AST patch grid (e.g. 12x100 for 10 s). (The paper does not describe the AudioSet-checkpoint-to-shorter-input case; the authors' *code* centre-crops the time axis, which was checked in the source on GitHub.)
* **Datasets and results:** Table 1, AudioSet: AST single model **0.459 +/- 0.000** mAP; Ensemble-S 0.475; Ensemble-M **0.485**. Table 7: ESC-50 AST-S (ImageNet) **88.7 +/- 0.7**, AST-P (ImageNet + AudioSet) **95.6 +/- 0.4**; Speech Commands V2 (35 classes) AST-S **98.11 +/- 0.05**, AST-P 97.88 +/- 0.03. Table 3: DeiT with distillation (87M parameters, 0.852 ImageNet top-1) is the initialisation used and gives the best AudioSet mAP. ESC-50 training: frequency/time masking, batch size 48, Adam, 20 epochs, initial lr 1e-5 for AST-P, 5-fold CV repeated three times.
* **Notes:** the HuggingFace checkpoint name `ast-finetuned-audioset-10-10-0.4593` encodes strides 10/10 and mAP 0.4593 (consistent with the single-model 0.459). The HuggingFace extractor uses Kaldi window type "hanning", whereas the paper text says Hamming (minor, not investigated further).
* **Verdict:** VERIFIED (with the qualifiers in the summary table).

### [5] E. Çakır, G. Parascandolo, T. Heittola, H. Huttunen, T. Virtanen, "Convolutional Recurrent Neural Networks for Polyphonic Sound Event Detection"
* **Venue (Crossref):** IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 25, no. 6, pp. 1291-1303, 2017. DOI 10.1109/TASLP.2017.2690575. arXiv 1702.06286. Link: https://arxiv.org/abs/1702.06286
* **Method:** convolutional layers (filters 5x5) with batch normalisation after each, pooling over frequency, then recurrent layers, then a feed-forward output with sigmoid; log mel band energies as input; binary cross-entropy; Adam. **GRU choice (Section III):** "in recurrent layers we opted for GRU, since preliminary experiments using LSTM yielded similar results and GRU units have a smaller number of parameters". Task is **polyphonic** (multi-label, frame-wise) event detection, unlike single-label clip classification here.
* **Datasets (Section III-A):** TUT-SED Synthetic 2016, TUT-SED 2009, TUT-SED 2016 (development), CHiME-Home.
* **Key result (Section IV):** "the CRNNs consistently outperforms CNNs, RNNs and the two baseline methods on all three datasets for the main metric" (e.g. on TUT-SED Synthetic 2016, absolute frame-based F1 gains of 6.6 % over the CNN and 13.6 % over the RNN).
* **Verdict:** VERIFIED, with the dataset-list difference noted.

### [6] A. Howard et al., "Searching for MobileNetV3"
* **Venue (Crossref):** 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1314-1324. DOI 10.1109/ICCV.2019.00140. arXiv 1905.02244 (v5, 20 Nov 2019). Links: https://arxiv.org/abs/1905.02244 ; https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf
* **Method:** hardware-aware network architecture search plus the NetAdapt algorithm, then architecture improvements (new nonlinearities such as h-swish, squeeze-and-excite style attention from MnasNet, a redesigned last stage); two models, MobileNetV3-Large and -Small. Latency measured on a Pixel phone using a single large core.
* **Datasets:** ImageNet classification, COCO detection, Cityscapes segmentation (new decoder LR-ASPP).
* **Abstract of the ICCV version and arXiv v5 (checked by me):** "MobileNetV3-Large is 3.2% more accurate on ImageNet classification while reducing latency by 20% compared to MobileNetV2. MobileNetV3-Small is 6.6% more accurate compared to a MobileNetV2 model with comparable latency." **arXiv v1 abstract (checked):** 15 % latency reduction for Large, and Small "4.6% more accurate while reducing latency by 5%". The numbers were revised, so cite the camera-ready.
* **Verdict:** VERIFIED against the final version.

### [7] B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"
* **Venue (Crossref):** 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2704-2713. DOI 10.1109/CVPR.2018.00286. arXiv 1712.05877 (v1, 15 Dec 2017). Link: https://arxiv.org/abs/1712.05877
* **Method:** an affine 8-bit quantization scheme (weights and activations) with integer-only inference arithmetic (32-bit accumulators), plus a training procedure that simulates quantization during training so accuracy is preserved ("quantization-aware training").
* **Datasets / hardware:** ImageNet classification and COCO detection; MobileNet latency-vs-accuracy compared against float baselines on Qualcomm Snapdragon 835 LITTLE cores (Fig. 1.1); implementations for ARM NEON and x86 mentioned; Qualcomm Hexagon as an integer-only target.
* **Verdict:** VERIFIED. This is QAT; our compression is post-training (no retraining), so for PTQ cite a PTQ-specific source (see section 3).

### [8] Q. Kong, Y. Cao, T. Iqbal, Y. Wang, W. Wang, M. D. Plumbley, "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition"
* **Venue (Crossref):** IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, pp. 2880-2894, 2020. DOI 10.1109/TASLP.2020.3030497. arXiv 1912.10211 (v5). Link: https://arxiv.org/abs/1912.10211
* **Method:** CNN families (CNN6/10/14, ResNet, MobileNet, and the proposed Wavegram-Logmel-CNN taking waveform and log-mel) pretrained on AudioSet and transferred.
* **Key results:** abstract: "Our best PANN system achieves a state-of-the-art mean average precision (mAP) of 0.439 on AudioSet tagging, outperforming the best previous system of 0.392" - the 0.439 is the **Wavegram-Logmel-CNN**. Table IV: **CNN14 0.431**. Transfer (Section VI-D lists five datasets: ESC-50, DCASE 2019 Task 1, DCASE 2018 Task 2, MSoS, GTZAN; the abstract says "six" tasks, probably counting AudioSet tagging itself, which I could not confirm): ESC-50 accuracy 0.947 for the best system. UrbanSound8K is not a transfer task there (it is only cited in the related-work text).
* **Verdict:** DIFFERS from the old table (wrong model for 0.439).

### [9] D. S. Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
* **Venue (arXiv journal reference + Crossref):** Proc. Interspeech 2019, pp. 2613-2617. DOI 10.21437/Interspeech.2019-2680. arXiv 1904.08779 (v3). Link: https://arxiv.org/abs/1904.08779
* **Method:** augmentation applied directly to the log-mel filter bank features: time warping, frequency masking, time masking; applied to Listen, Attend and Spell networks.
* **Key results:** LibriSpeech 960h: 6.8 % WER on test-other without a language model, 5.8 % with shallow fusion (previous best hybrid 7.5 %); Switchboard 300h: 7.2 %/14.6 % (no LM) and 6.8 %/14.1 % (fusion) vs 8.3 %/17.3 %.
* **Verdict:** VERIFIED.

### [10] S. Han, H. Mao, W. J. Dally, "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding"
* **Venue (arXiv comment):** ICLR 2016 (oral). arXiv 1510.00149 (v5). Link: https://arxiv.org/abs/1510.00149
* **Method:** three stages: prune small-magnitude connections and retrain; quantize weights to a shared code book (32 to 5 bits per connection) and retrain; Huffman-code the result.
* **Key results (abstract, Table 1):** storage reduced "by 35x to 49x without affecting their accuracy"; pruning alone cuts connections by 9x to 13x; AlexNet 240 MB to 6.9 MB (35x), VGG-16 552 MB to 11.3 MB (49x), LeNet-300-100 40x and LeNet-5 39x on MNIST; the paper also reports 3x-4x layer-wise speedup and 3x-7x energy efficiency on CPU, GPU and mobile GPU.
* **Verdict:** VERIFIED. For this project: pruning-only, unstructured, no sparse kernels, so only compressed size and non-zero counts change.

### Other references checked for existence (metadata from arXiv API / Crossref)
* Dataset README (local): see [1].
* All 24 arXiv identifiers listed in section 3 below returned the expected title and authors from the arXiv API.

---

## 3. Suggested additional references (to reach 12+ for the final report)

The ten papers above already give 10 distinct references. The report needs 12+ with an integrated related-work section. All entries below exist (title/authors from arXiv API; venue/pages from Crossref
where listed). "Venue from memory" means the arXiv record did not state the venue, so confirm it on the publisher page before citing.

**Strongly recommended (each is needed to explain something this project does):**

| Paper | Why add it | Check status |
|---|---|---|
| J. F. Gemmeke et al., "Audio Set: An ontology and human-labeled dataset for audio events", ICASSP 2017, pp. 776-780 (DOI 10.1109/ICASSP.2017.7952261) | the pretraining data of the AST checkpoint; needed for the AudioSet-overlap limitation | Crossref verified |
| H. Touvron et al., "Training data-efficient image transformers & distillation through attention" (DeiT), arXiv 2012.12877 | AST is initialised from DeiT (AST paper Table 3), so the lineage ImageNet -> AudioSet needs it | arXiv verified; venue (ICML 2021) from memory |
| R. Krishnamoorthi, "Quantizing deep convolutional networks for efficient inference: A whitepaper", arXiv 1806.08342 | defines post-training dynamic/static quantization and calibration, which Jacob et al. (QAT) does not | arXiv verified (preprint only) |
| D. Blalock et al., "What is the State of Neural Network Pruning?", MLSys 2020, arXiv 2003.03033 | reviews magnitude pruning, sparsity vs real speedups, evaluation pitfalls | arXiv verified (venue from arXiv comment) |
| K. Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder ...", EMNLP 2014, arXiv 1406.1078 | introduces the GRU used in the CRNN | arXiv verified (venue from arXiv comment) |

**Useful for the methodology section:**

| Paper | Why | Check status |
|---|---|---|
| A. Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT), arXiv 2010.11929 | patch embedding and transformer encoder that AST adapts | arXiv verified; venue (ICLR 2021) from memory |
| A. Vaswani et al., "Attention Is All You Need", arXiv 1706.03762 | self-attention itself | arXiv verified; venue (NeurIPS 2017) from memory |
| S. Ioffe, C. Szegedy, "Batch Normalization", arXiv 1502.03167 | used in all CNN-family models | arXiv verified; venue (ICML 2015) from memory |
| I. Loshchilov, F. Hutter, "Decoupled Weight Decay Regularization" (AdamW), ICLR 2019, arXiv 1711.05101 | the optimiser | arXiv verified (venue from arXiv comment) |
| I. Loshchilov, F. Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", ICLR 2017, arXiv 1608.03983 | origin of cosine annealing | arXiv verified (venue from arXiv comment); note SGDR uses restarts, we use a single cosine |
| C. Szegedy et al., "Rethinking the Inception Architecture for Computer Vision", CVPR 2016, pp. 2818-2826 | introduces label smoothing | Crossref verified |
| R. Müller, S. Kornblith, G. Hinton, "When Does Label Smoothing Help?", NeurIPS 2019, arXiv 1906.02629 | analysis of label smoothing | arXiv verified (venue from arXiv comment) |
| M. Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018, pp. 4510-4520 | inverted residual block that MobileNetV3 builds on | Crossref verified |
| S. Hershey et al., "CNN Architectures for Large-Scale Audio Classification", ICASSP 2017, pp. 131-135 | audio CNNs on log-mel at AudioSet scale | Crossref verified |
| J. Chung et al., "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling", arXiv 1412.3555 | GRU vs LSTM comparison | arXiv verified (NIPS 2014 workshop per arXiv comment) |
| M. Nagel et al., "A White Paper on Neural Network Quantization", arXiv 2106.08295 | modern PTQ/QAT overview | arXiv verified (preprint) |
| M. Zhu, S. Gupta, "To prune, or not to prune", arXiv 1710.01878 | magnitude pruning trade-offs | arXiv verified; venue (ICLR 2018 workshop) from memory |

**Not verified / not recommended without checking:** I did not look for UrbanSound8K-specific papers beyond the ones above (e.g. later results on this dataset); if you want to compare your numbers with
the literature, find those yourself and verify their protocol (many use the 10-fold CV differently or add augmentation). I did not verify the AudioSet challenge baselines other than Hershey et al.

---

## 4. Verification log (reproducible)

* arXiv API, 24 identifiers: 1608.04363, 2104.01778, 1702.06286, 1905.02244, 1712.05877, 1912.10211, 1904.08779, 1510.00149, 2010.11929, 2012.12877, 1502.03167, 1406.1078, 1512.00567, 1906.02629, 1711.05101,
  1608.03983, 1801.04381, 1609.09430, 1806.08342, 2106.08295, 1710.01878, 2003.03033, 1706.03762, 1412.3555 - all returned the expected papers.
* PDFs read: arXiv versions of papers 2, 4, 5, 6 (v1 and v5), 7, 8, 9, 10; ICCV open-access PDF of paper 6; Piczak's PDF (3). Not read: Salamon et al. 2014 (inaccessible).
* Crossref confirmed venue/volume/pages/year for papers 1-10 and for AudioSet, Hershey, MobileNetV2 and the Inception paper.
