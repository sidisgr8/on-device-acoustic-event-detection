# Generated model summaries (scripts/describe_models.py; do not edit by hand)

Shapes are for one clip (batch 1). MAC counts include only Conv2d, Linear and GRU layers (no BatchNorm, activations, pooling, softmax; for AST the attention matmuls are added separately).

## CNN baseline (4 x [conv3x3-BN-ReLU-maxpool2], GAP, dropout 0.3, linear)

| layer | type | output shape (B=1) | params |
|---|---|---|---|
| features.0 | Sequential | (1, 32, 64, 100) | 352 |
| features.1 | Sequential | (1, 64, 32, 50) | 18,560 |
| features.2 | Sequential | (1, 128, 16, 25) | 73,984 |
| features.3 | Sequential | (1, 256, 8, 12) | 295,424 |
| global average pool (mean over F,T) | ops | (1, 256) | 0 |
| head.1 (Linear) | Linear | (1, 10) | 2,570 |

**Total parameters: 390,890** (trainable). Conv/Linear/GRU multiply-accumulates per clip: 361.3 M (conv 361.3 M, linear 0.0 M, GRU 0.0 M).

Per block: Conv2d(no bias) -> BatchNorm2d -> ReLU -> MaxPool2d(2). Channels 1->32->64->128->256.
- features.0: conv 288 + BN 64 = 352 params
- features.1: conv 18,432 + BN 128 = 18,560 params
- features.2: conv 73,728 + BN 256 = 73,984 params
- features.3: conv 294,912 + BN 512 = 295,424 params

## MobileNetV3-Small (timm `mobilenetv3_small_100`, 1-channel stem, 10-way classifier)

| layer | type | output shape (B=1) | params |
|---|---|---|---|
| conv_stem | Conv2d | (1, 16, 64, 101) | 144 |
| bn1 (BN+hardswish) | BatchNormAct2d | (1, 16, 64, 101) | 32 |
| blocks.0 | Sequential | (1, 16, 32, 51) | 744 |
| blocks.1 | Sequential | (1, 24, 16, 26) | 9,280 |
| blocks.2 | Sequential | (1, 40, 8, 13) | 128,264 |
| blocks.3 | Sequential | (1, 48, 8, 13) | 51,768 |
| blocks.4 | Sequential | (1, 96, 4, 7) | 680,040 |
| blocks.5 | Sequential | (1, 576, 4, 7) | 56,448 |
| global_pool | SelectAdaptivePool2d | (1, 576, 1, 1) | 0 |
| conv_head (1x1) | Conv2d | (1, 1024, 1, 1) | 590,848 |
| classifier | Linear | (1, 10) | 10,250 |

**Total parameters: 1,527,818** (trainable). Conv/Linear/GRU multiply-accumulates per clip: 28.6 M (conv 28.6 M, linear 0.0 M, GRU 0.0 M).

Blocks are inverted-residual (depthwise-separable) stages; several use squeeze-excite and hard-swish. Stage module counts: [1, 2, 3, 2, 3, 1]. Head: 1x1 conv to 1024 channels, hard-swish, linear 10. Head dropout (timm drop_rate): 0.0.
Stem weight shape (out,in,kh,kw) = (16, 1, 3, 3) (1 input channel); ImageNet run: timm sums the pretrained RGB filters into this single-channel filter.

## CRNN (3 conv blocks -> linear 128 -> BiGRU 128 -> [mean;max] over time -> dropout 0.3 -> linear)

| layer | type | output shape (B=1) | params |
|---|---|---|---|
| convs.0 | Sequential | (1, 32, 64, 100) | 352 |
| convs.1 | Sequential | (1, 64, 32, 50) | 18,560 |
| convs.2 | Sequential | (1, 128, 16, 50) | 73,984 |
| proj (Linear 2048->128) | Linear | (1, 50, 128) | 262,272 |
| gru (BiGRU, hidden 128) | GRU | (1, 50, 256) | 198,144 |
| head.1 (Linear 512->10) | Linear | (1, 10) | 5,130 |

**Total parameters: 558,442** (trainable). Conv/Linear/GRU multiply-accumulates per clip: 266.3 M (conv 243.3 M, linear 13.1 M, GRU 9.8 M).

Shapes through the network (from `CRNN.forward(return_shapes=True)`):

- input: (1, 1, 128, 201)
- after_conv: (1, 128, 16, 50)
- flattened_per_step: (1, 50, 2048)
- projected: (1, 50, 128)
- gru_out: (1, 50, 256)
- pooled: (1, 512)

GRU parameters: 198,144 (2 directions x 3 gates x (128x128 input + 128x128 recurrent + 2 biases)).

## AST (HF `ASTForAudioClassification`, 400-frame input, AudioSet checkpoint architecture)

| layer | type | output shape (B=1) | params |
|---|---|---|---|
| embeddings.patch_embeddings.projection (Conv2d 16x16, stride 10) | Conv2d | (1, 768, 12, 39) | 197,376 |
| embeddings (patch + cls/dist + pos) | ASTEmbeddings | (1, 470, 768) | 559,872 |
| layers.0 (transformer block) | ASTLayer | (1, 470, 768) | 7,087,872 |
| layers.11 (transformer block) | ASTLayer | (1, 470, 768) | 7,087,872 |
| layernorm | LayerNorm | (1, 470, 768) | 1,536 |
| classifier.layernorm | LayerNorm | (1, 768) | 1,536 |
| classifier.dense (Linear 768->10) | Linear | (1, 10) | 7,690 |

**Total parameters: 85,625,098** (trainable). Conv/Linear/GRU multiply-accumulates per clip: 40,011.3 M (conv 92.0 M, linear 39,919.3 M, GRU 0.0 M).

Tokens: 2 special + 12 (freq) x 39 (time) = 470. Attention score+value matmul MACs (analytic, 12 layers x 2 x T^2 x D): 4,071.6 M; so total ~ 44,082.9 M MACs per clip.
Config: hidden 768, layers 12, heads 12, MLP 3072, patch 16, strides (freq 10, time 10), max_length 400, mel bins 128, hidden dropout 0.0, attention dropout 0.0.
Parameter groups: patch conv 197,376; position embeddings 360,960 (1, 470, 768); cls+distillation tokens 1,536; one transformer block 7,087,872 (x12 = 85,054,464); final layernorm 1,536; classifier 9,226 (LayerNorm 1,536 + dense 7,690).
