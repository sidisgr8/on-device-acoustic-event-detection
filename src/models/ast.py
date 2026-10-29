"""Model 3: Audio Spectrogram Transformer fine-tuned from the AudioSet checkpoint, 400-frame input.

The public checkpoint is built for 1024 frames. HuggingFace would re-initialise the position
embeddings for a different `max_length`, discarding what was pretrained, so they are cropped
by hand following the AST authors' code (YuanGongND/ast, src/models/ast_models.py, audioset
branch): reshape the 1212 patch embeddings to the (12 freq x 101 time) grid, keep all 12
frequency rows, take the time columns [50 - t/2, 50 - t/2 + t) (a CENTRED crop, t = 39 for 400
frames), flatten back, and prepend the two special-token embeddings (cls, distillation).
`pos_crop="first"` (first t columns) is available but not the default (see docs/DECISIONS.md D-08).
"""
import copy

import torch
from torch import nn
from transformers import ASTConfig, ASTForAudioClassification

CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
POS_KEY = "audio_spectrogram_transformer.embeddings.position_embeddings"
N_SPECIAL = 2  # cls + distillation tokens


def patch_grid(cfg: ASTConfig):
    f = (cfg.num_mel_bins - cfg.patch_size) // cfg.frequency_stride + 1
    t = (cfg.max_length - cfg.patch_size) // cfg.time_stride + 1
    return f, t


def crop_position_embeddings(pos: torch.Tensor, src_grid, dst_grid, pos_crop="center"):
    """pos: (1, 2 + F*T, D) from the checkpoint -> (1, 2 + F*T', D). Returns (new, info)."""
    (f, t_src), (f2, t_dst) = src_grid, dst_grid
    assert f == f2, "frequency axis must be unchanged"
    assert t_dst <= t_src, "only cropping is implemented"
    d = pos.shape[-1]
    assert pos.shape[1] == N_SPECIAL + f * t_src, (pos.shape, f, t_src)
    special, patches = pos[:, :N_SPECIAL], pos[:, N_SPECIAL:]
    # token order is frequency-major (index = f_idx * T + t_idx), as produced by the patch conv
    grid = patches.reshape(1, f, t_src, d)
    start = t_src // 2 - t_dst // 2 if pos_crop == "center" else 0
    new = grid[:, :, start:start + t_dst, :].reshape(1, f * t_dst, d)
    info = {"src_shape": tuple(pos.shape), "src_grid_FxT": (f, t_src), "dst_grid_FxT": (f, t_dst),
            "pos_crop": pos_crop, "time_slice": (start, start + t_dst),
            "dst_shape": (1, N_SPECIAL + f * t_dst, d)}
    return torch.cat([special, new], dim=1), info


class ASTClassifier(nn.Module):
    """Input: normalised fbank (B, 400, 128). Output: logits (B, n_classes)."""

    def __init__(self, n_classes=10, max_length=400, checkpoint=CHECKPOINT, pos_crop="center",
                 pretrained=True):
        super().__init__()
        src_cfg = ASTConfig.from_pretrained(checkpoint)
        cfg = copy.deepcopy(src_cfg)
        cfg.max_length = max_length
        cfg.num_labels = n_classes
        cfg.id2label = {i: str(i) for i in range(n_classes)}
        cfg.label2id = {v: k for k, v in cfg.id2label.items()}
        self.model = ASTForAudioClassification(cfg)
        self.provenance = {"pretrained": pretrained, "checkpoint": checkpoint}
        if pretrained:
            src = ASTForAudioClassification.from_pretrained(checkpoint)
            sd = {k: v for k, v in src.state_dict().items()
                  if k != POS_KEY and not k.startswith("classifier.dense")}  # new 10-way head
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            assert set(missing) == {POS_KEY, "classifier.dense.weight", "classifier.dense.bias"}, missing
            assert not unexpected, unexpected
            new_pos, info = crop_position_embeddings(
                src.state_dict()[POS_KEY], patch_grid(src_cfg), patch_grid(cfg), pos_crop)
            with torch.no_grad():
                self.model.state_dict()[POS_KEY].copy_(new_pos)
            self.provenance.update(info)
            self.provenance["head"] = "classifier.dense re-initialised (10-way)"
            del src

    def forward(self, x):
        return self.model(input_values=x).logits
