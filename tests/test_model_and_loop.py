import json

import numpy as np
import pytest
import torch

from src.models import build_model
from src.train.loop import run_round


def test_cnn_forward_shape_and_params():
    m = build_model({"name": "cnn_baseline", "dropout": 0.3})
    out = m(torch.randn(3, 1, 128, 201))
    assert out.shape == (3, 10)
    n = sum(p.numel() for p in m.parameters())
    assert 300_000 < n < 500_000


def tiny_cfg(tmp, max_epochs=3):
    return {"protocol_version": 2, "experiment": "t", "model": {"name": "cnn_baseline", "dropout": 0.0, "channels": (4, 4, 4, 4)},
            "data": {"cache": str(tmp), "floor_db": -80.0},
            "train": {"lr": 1e-3, "weight_decay": 0.01, "batch_size": 8, "max_epochs": max_epochs,
                      "warmup_epochs": 1, "early_stopping": False, "label_smoothing": 0.1, "amp": "none",
                      "seed": 42},
            "specaugment": {"freq_masks": 1, "freq_width": 4, "time_masks": 1, "time_width": 4}}


def test_round_runs_writes_results_and_evaluates_test_once(synthetic_store, tmp_path):
    cfg = tiny_cfg(tmp_path)
    res = run_round(cfg, 2, tmp_path / "res", tmp_path / "ck", "cpu", synthetic_store, log=lambda *_: None)
    p = tmp_path / "res" / "t" / "fold2.json"
    assert p.exists()
    r = json.loads(p.read_text())
    assert r["n_test"] == 12 and r["fold"] == 2
    assert np.array(r["confusion_matrix"]).sum() == 12
    assert {"accuracy", "macro_f1", "train_time_s", "by_padding"} <= set(r)
    # second call must not retrain or re-evaluate the test fold
    mtime = p.stat().st_mtime_ns
    again = run_round(cfg, 2, tmp_path / "res", tmp_path / "ck", "cpu", synthetic_store, log=lambda *_: None)
    assert again["accuracy"] == res["accuracy"] and p.stat().st_mtime_ns == mtime


def test_resume_continues_from_last_checkpoint(synthetic_store, tmp_path):
    cfg = tiny_cfg(tmp_path, max_epochs=4)
    # Run 2 epochs, "crash", then rerun with the full budget and compare with an uninterrupted run.
    cfg2 = tiny_cfg(tmp_path, max_epochs=4)
    cfg2["train"]["max_epochs"] = 4
    import src.train.loop as L
    # simulate interruption by capping epochs via an exception after epoch 2
    calls = {"n": 0}
    orig = L.atomic_save

    def crash_after_two(obj, path):
        orig(obj, path)
        if path.name == "last.pt":
            calls["n"] += 1
            if calls["n"] == 2:
                raise KeyboardInterrupt

    L.atomic_save = crash_after_two
    try:
        with pytest.raises(KeyboardInterrupt):
            run_round(cfg, 3, tmp_path / "a", tmp_path / "ca", "cpu", synthetic_store, log=lambda *_: None)
    finally:
        L.atomic_save = orig
    resumed = run_round(cfg, 3, tmp_path / "a", tmp_path / "ca", "cpu", synthetic_store, log=lambda *_: None)
    straight = run_round(cfg2, 3, tmp_path / "b", tmp_path / "cb", "cpu", synthetic_store, log=lambda *_: None)
    assert resumed["epochs_run"] == straight["epochs_run"] == 4
    assert [h["epoch"] for h in resumed["history"]] == [1, 2, 3, 4]
    np.testing.assert_allclose([h["train_loss"] for h in resumed["history"]],
                               [h["train_loss"] for h in straight["history"]], rtol=1e-4)


def test_protocol_guards(synthetic_store, tmp_path):
    cfg = tiny_cfg(tmp_path)
    bad = {k: v for k, v in cfg.items() if k != "protocol_version"}
    with pytest.raises(ValueError, match="protocol_version"):
        run_round(bad, 2, tmp_path / "r", tmp_path / "c", "cpu", synthetic_store, log=lambda *_: None)
    cfg2 = tiny_cfg(tmp_path)
    del cfg2["train"]["early_stopping"]
    with pytest.raises(ValueError, match="early_stopping"):
        run_round(cfg2, 2, tmp_path / "r", tmp_path / "c", "cpu", synthetic_store, log=lambda *_: None)
    cfg3 = tiny_cfg(tmp_path)
    cfg3["train"]["amp"] = "auto"      # must be resolved by the queue first
    with pytest.raises(ValueError, match="amp"):
        run_round(cfg3, 2, tmp_path / "r", tmp_path / "c", "cpu", synthetic_store, log=lambda *_: None)


def test_fixed_epochs_without_early_stopping_and_best_checkpoint_kept(synthetic_store, tmp_path):
    cfg = tiny_cfg(tmp_path, max_epochs=4)
    cfg["train"]["lr"] = 0.0           # validation accuracy cannot improve after epoch 1
    r = run_round(cfg, 2, tmp_path / "res", tmp_path / "ck", "cpu", synthetic_store, log=lambda *_: None)
    assert r["epochs_run"] == 4 and not r["stopped_early"] and r["early_stopping"] is False
    assert r["protocol_version"] == 2 and r["config_hash"]
    ck = tmp_path / "ck" / "t" / "fold2"
    assert (ck / "best.pt").exists() and (ck / "best.json").exists()
    assert not (ck / "last.pt").exists()                      # resume file removed, best.pt kept
    # contrast: the same run WITH early stopping (patience 1) ends early
    cfg_es = tiny_cfg(tmp_path, max_epochs=4)
    cfg_es["experiment"] = "t_es"
    cfg_es["train"].update(lr=0.0, early_stopping=True, patience=1)
    r2 = run_round(cfg_es, 2, tmp_path / "res", tmp_path / "ck", "cpu", synthetic_store, log=lambda *_: None)
    assert r2["epochs_run"] == 2 and r2["stopped_early"]


def test_cosine_decays_over_exactly_max_epochs(synthetic_store, tmp_path):
    cfg = tiny_cfg(tmp_path, max_epochs=4)
    r = run_round(cfg, 2, tmp_path / "res", tmp_path / "ck", "cpu", synthetic_store, log=lambda *_: None)
    lrs = [h["lr"] for h in r["history"]]
    assert lrs[0] < lrs[1] or lrs[0] <= 1e-3         # warm-up then decay
    assert lrs[-1] < 1e-6                              # fully decayed at the end of the last epoch
