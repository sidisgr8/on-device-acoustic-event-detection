import torch

from src.data.dataset import SpecAugment


def make(**kw):
    return SpecAugment(freq_masks=2, freq_width=16, time_masks=2, time_width=20,
                       generator=torch.Generator().manual_seed(0), **kw)


def test_identity_in_eval_mode():
    aug = make().eval()
    x = torch.randn(8, 1, 128, 201)
    assert torch.equal(aug(x), x)


def test_modifies_only_in_train_mode_and_within_limits():
    aug = make().train()
    x = torch.ones(64, 1, 128, 201)
    y = aug(x)
    assert not torch.equal(x, y)
    assert set(y.unique().tolist()) <= {0.0, 1.0}
    masked_rows = (y == 0).all(dim=3).sum(dim=2).squeeze()   # fully masked frequency bins
    masked_cols = (y == 0).all(dim=2).sum(dim=2).squeeze()   # fully masked time frames
    assert (masked_rows <= 2 * 16).all() and (masked_cols <= 2 * 20).all()
    assert masked_rows.max() > 0 and masked_cols.max() > 0


def test_input_not_modified_in_place():
    aug = make().train()
    x = torch.ones(4, 1, 128, 201)
    aug(x)
    assert (x == 1).all()


def test_reproducible_with_same_generator_seed():
    a, b = make().train(), make().train()
    x = torch.ones(16, 1, 128, 201)
    assert torch.equal(a(x), b(x))


def test_train_loop_toggles_augmentation_off_for_eval(synthetic_store, tmp_path):
    """The validation/test path must never see augmentation: evaluate() does not use `aug`."""
    import inspect
    from src.train import loop
    assert "aug" not in inspect.signature(loop.evaluate).parameters
    assert "aug(" not in inspect.getsource(loop.evaluate)


# ---- D-36: identical augmentation in SECONDS for AST and the CNN family -------------------------------------
import pytest
import yaml
from pathlib import Path

CNN_FAMILY = ["cnn_baseline", "mobilenetv3_small_scratch", "mobilenetv3_small_imagenet", "crnn"]
CFG_DIR = Path(__file__).resolve().parents[1] / "configs"


def sa_cfg(name):
    return yaml.safe_load(open(CFG_DIR / f"{name}.yaml"))["specaugment"]


def test_time_mask_is_40_frames_for_ast_and_20_for_the_cnn_family():
    for name in CNN_FAMILY:
        assert sa_cfg(name)["time_width"] == 20, name
    assert sa_cfg("ast")["time_width"] == 40


def test_masks_cover_the_same_duration_in_seconds():
    from src.features import params as P
    cnn_hop_s = P.HOP / P.SR                 # 20 ms per log-mel frame
    ast_shift_s = 10 / 1000                  # 10 ms Kaldi frame shift
    assert sa_cfg("cnn_baseline")["time_width"] * cnn_hop_s == pytest.approx(0.4)
    assert sa_cfg("ast")["time_width"] * ast_shift_s == pytest.approx(0.4)


def test_everything_else_in_the_augmentation_is_identical_across_models():
    ref = {k: v for k, v in sa_cfg("cnn_baseline").items() if k != "time_width"}
    for name in CNN_FAMILY[1:] + ["ast"]:
        assert {k: v for k, v in sa_cfg(name).items() if k != "time_width"} == ref, name
    assert ref == {"freq_masks": 2, "freq_width": 16, "time_masks": 2}


def test_realised_masks_respect_each_models_width():
    torch.manual_seed(0)
    ast = SpecAugment(**sa_cfg("ast"), freq_dim=2, time_dim=1, generator=torch.Generator().manual_seed(1)).train()
    cnn = SpecAugment(**sa_cfg("cnn_baseline"), generator=torch.Generator().manual_seed(1)).train()
    y_ast = ast(torch.ones(256, 400, 128))
    y_cnn = cnn(torch.ones(256, 1, 128, 201))
    t_ast = (y_ast == 0).all(dim=2).sum(dim=1)                 # fully masked time frames per clip (AST layout B,T,F)
    t_cnn = (y_cnn == 0).all(dim=2).sum(dim=2).squeeze()       # (CNN layout B,1,F,T)
    assert t_ast.max() <= 2 * 40 and t_ast.max() > 2 * 20      # AST reaches beyond what a 20-frame mask could
    assert t_cnn.max() <= 2 * 20
    f_ast = (y_ast == 0).all(dim=1).sum(dim=1)
    assert f_ast.max() <= 2 * 16
