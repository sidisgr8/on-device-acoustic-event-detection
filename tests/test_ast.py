import numpy as np
import pytest
import torch

from src.data.dataset import SpecAugment

pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def ast_model():
    from src.models import build_model
    torch.set_num_threads(2)
    try:
        return build_model({"name": "ast", "pretrained": True})
    except Exception as e:  # offline
        pytest.skip(f"AST checkpoint unavailable: {e}")


def test_position_embedding_crop_shapes_and_values(ast_model):
    from transformers import ASTForAudioClassification
    from src.models.ast import CHECKPOINT, POS_KEY
    prov = ast_model.provenance
    assert prov["src_grid_FxT"] == (12, 101) and prov["dst_grid_FxT"] == (12, 39)
    assert prov["dst_shape"] == (1, 470, 768) and prov["time_slice"] == (31, 70)
    new = ast_model.model.state_dict()[POS_KEY]
    assert new.shape == (1, 470, 768)
    src = ASTForAudioClassification.from_pretrained(CHECKPOINT).state_dict()[POS_KEY]
    assert torch.equal(new[:, :2], src[:, :2])                        # cls + distillation tokens
    grid_src = src[:, 2:].reshape(1, 12, 101, 768)
    grid_new = new[:, 2:].reshape(1, 12, 39, 768)
    assert torch.equal(grid_new, grid_src[:, :, 31:70])               # all 12 freq rows, centred time crop
    # token order really is frequency-major (checked against the model's own patchifier)
    emb = ast_model.model.audio_spectrogram_transformer.embeddings
    assert emb.get_shape(ast_model.model.config) == (12, 39)


def test_forward_and_backward_finite(ast_model):
    x = torch.randn(2, 400, 128)
    out = ast_model(x)
    assert out.shape == (2, 10)
    loss = torch.nn.functional.cross_entropy(out, torch.tensor([0, 1]))
    loss.backward()
    assert torch.isfinite(loss)
    assert sum(p.numel() for p in ast_model.parameters()) == 85_625_098


def test_pretrained_weights_actually_loaded(ast_model):
    from transformers import ASTForAudioClassification
    from src.models.ast import CHECKPOINT
    src = ASTForAudioClassification.from_pretrained(CHECKPOINT).state_dict()
    mine = ast_model.model.state_dict()
    skip = {"audio_spectrogram_transformer.embeddings.position_embeddings",
            "classifier.dense.weight", "classifier.dense.bias"}
    checked = 0
    for k, v in src.items():
        if k in skip:
            continue
        assert torch.equal(mine[k], v), k
        checked += 1
    assert checked == 200
    assert not torch.equal(mine["classifier.dense.weight"][:1], src["classifier.dense.weight"][:1])


def test_specaugment_on_time_first_layout():
    aug = SpecAugment(freq_dim=2, time_dim=1, generator=torch.Generator().manual_seed(0)).train()
    y = aug(torch.ones(32, 400, 128))
    rows = (y == 0).all(dim=1).sum(dim=1)   # fully masked frequency bins per clip
    cols = (y == 0).all(dim=2).sum(dim=1)   # fully masked time frames per clip
    assert (rows <= 32).all() and (cols <= 40).all() and rows.max() > 0 and cols.max() > 0
    assert torch.equal(SpecAugment(freq_dim=2, time_dim=1).eval()(torch.ones(2, 400, 128)), torch.ones(2, 400, 128))


def test_ast_features_normalised_with_checkpoint_constants(real_cache):
    from src.data.dataset import FeatureStore, make_round_ast
    store = FeatureStore()
    tr, va, te, st = make_round_ast(store, 1)
    assert tr.x.shape[1:] == (400, 128) and st.mean == pytest.approx(-4.2677393)
    assert st.std == pytest.approx(2 * 4.5689974)
    assert set(store.fold[tr.rows.numpy()]) == set(range(3, 11))      # same split rule as the CNN family
    assert set(store.fold[va.rows.numpy()]) == {2} and set(store.fold[te.rows.numpy()]) == {1}
    # sanity: normalised features are roughly standard (AudioSet constants on this data)
    v = tr.x[:200]
    assert abs(v.mean().item()) < 1.0 and 0.3 < v.std().item() < 3.0


def test_patch_token_order_is_frequency_major():
    """The position-embedding crop reshapes the 1212 patch embeddings to (12 freq, 101 time). That is only
    right if token index = freq_idx * T + time_idx. An impulse at (time frame 5, mel bin 125) lies only in
    time-patch 0 and freq-patch 11, so exactly token 11 * 39 + 0 = 429 must react (time-major would be 11)."""
    from src.models import build_model
    torch.set_num_threads(1)
    pe = build_model({"name": "ast", "pretrained": False}).model.audio_spectrogram_transformer \
        .embeddings.patch_embeddings.eval()
    base = pe(torch.zeros(1, 400, 128))
    x = torch.zeros(1, 400, 128)
    x[0, 5, 125] = 1.0
    changed = torch.nonzero((pe(x) - base).abs().sum(-1)[0] > 1e-6).flatten().tolist()
    assert changed == [11 * 39 + 0]
