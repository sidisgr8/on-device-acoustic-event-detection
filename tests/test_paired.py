import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.analysis import aggregate as A
from src.analysis.curves import load_curves, summarize
from src.analysis.paired import ablation_table, holm, paired_records, paired_stats, paired_table
from conftest import write_fold


def test_paired_stats_matches_scipy_and_known_values():
    a = np.array([0.80, 0.82, 0.78, 0.85, 0.81, 0.79, 0.83, 0.84, 0.80, 0.82])
    b = a - np.array([0.01, 0.02, 0.015, 0.03, 0.005, 0.02, 0.01, 0.025, 0.012, 0.018])
    s = paired_stats(a, b)
    d = a - b
    assert s["mean_diff"] == pytest.approx(d.mean()) and s["a_wins"] == 10 and s["ties"] == 0
    assert s["t_p"] == pytest.approx(stats.ttest_rel(a, b).pvalue)
    assert s["wilcoxon_p"] == pytest.approx(2 / 1024)                   # all 10 differences positive: exact p = 2/2^10
    assert s["ci_lo"] <= s["mean_diff"] <= s["ci_hi"] and s["ci_lo"] > 0
    assert paired_stats(a, b) == s                                       # fixed seed: deterministic


def test_antisymmetry_and_no_difference():
    a = np.array([0.7, 0.8, 0.75, 0.9, 0.6, 0.65])
    b = np.array([0.72, 0.78, 0.76, 0.88, 0.63, 0.64])
    s_ab, s_ba = paired_stats(a, b), paired_stats(b, a)
    assert s_ab["mean_diff"] == pytest.approx(-s_ba["mean_diff"])
    assert s_ab["wilcoxon_p"] == pytest.approx(s_ba["wilcoxon_p"]) and s_ab["t_p"] == pytest.approx(s_ba["t_p"])
    assert s_ab["ci_lo"] == pytest.approx(-s_ba["ci_hi"]) or abs(s_ab["ci_lo"] + s_ba["ci_hi"]) < 0.01   # bootstrap, same resamples
    same = paired_stats(a, a)
    assert same["mean_diff"] == 0 and same["wilcoxon_p"] == 1.0 and same["t_p"] == 1.0 and same["ties"] == 6


def test_constant_difference_has_zero_t_p():
    s = paired_stats(np.full(10, 0.8), np.full(10, 0.7))
    assert s["t_p"] == 0.0 and s["mean_diff"] == pytest.approx(0.1)


def test_holm_adjustment():
    adj = holm([0.01, 0.04, 0.03, 0.5])
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.5])                 # step-down, monotone, capped at 1
    assert holm([0.6, 0.7]) == pytest.approx([1.0, 1.0])


def _write(tmp, exp, accs, **kw):
    for k, acc in enumerate(accs, start=1):
        write_fold(tmp, exp, k, acc=acc, **kw)


def test_paired_table_covers_all_pairs_and_uses_common_folds(tmp_path):
    rng = np.random.default_rng(1)
    base = rng.uniform(0.6, 0.8, 10)
    _write(tmp_path, "cnn_baseline", base)
    _write(tmp_path, "crnn", base + 0.03 + rng.normal(0, 0.002, 10))
    for k in (1, 5, 9):                                                   # AST subset: only common folds are paired
        write_fold(tmp_path, "ast", k, acc=base[k - 1] + 0.1)
    recs = paired_records(tmp_path)
    assert len(recs) == 3
    ast_pair = [r for r in recs if r["b"] == "ast"][0]
    assert ast_pair["folds"] == [1, 5, 9] and ast_pair["n"] == 3
    df = paired_table(tmp_path)
    assert list(df["Folds"]) == ["10/10", "3/10", "3/10"]
    row = df[(df["A"] == "CNN baseline") & (df["B"] == "CRNN (CNN+BiGRU)")].iloc[0]
    assert abs(float(row["Mean diff A-B (pp)"]) + 3.0) < 0.3 and row["A ahead in"] == "0/10"
    assert paired_table(tmp_path, metric="macro_f1").shape[0] == 3


def test_ablation_table_only_declared_pairs_and_stays_out_of_main_rows(tmp_path):
    rng = np.random.default_rng(2)
    base = rng.uniform(0.5, 0.7, 10)
    _write(tmp_path, "mobilenetv3_small_scratch", base)
    _write(tmp_path, "mobilenetv3_small_imagenet", base + 0.15)
    _write(tmp_path, "a1_mobilenetv3_small_scratch_lr3e4", base + 0.10 + rng.normal(0, 0.005, 10))
    abl = ablation_table(tmp_path)
    assert len(abl) == 2 and set(abl["A"]) == {"[Ablation A-1] MobileNetV3-S (scratch, lr 3e-4)"}
    assert set(abl["B"]) == {"MobileNetV3-S (scratch)", "MobileNetV3-S (ImageNet)"}
    assert not any("A-1" in a or "A-1" in b for a, b in zip(paired_table(tmp_path)["A"], paired_table(tmp_path)["B"]))
    master = A.master_table(tmp_path, tmp_path / "ck")
    assert master["Model"].iloc[-1].startswith("[Ablation A-1]") and master["_ablation"].tolist() == [False, False, True]
    assert "a1_mobilenetv3_small_scratch_lr3e4" not in A.load_all(tmp_path)
    assert ablation_table(tmp_path / "nothing").empty


def test_curves_average_over_folds(tmp_path):
    d = tmp_path / "crnn"
    d.mkdir()
    for k, off in ((1, 0.0), (2, 0.2)):
        (d / f"fold{k}_epochs.jsonl").write_text("\n".join(json.dumps(
            {"epoch": e, "train_loss": 1 - 0.1 * e + off, "val_loss": 1.2 - 0.1 * e + off,
             "train_acc": 0.5 + 0.1 * e, "val_acc": 0.4 + 0.1 * e}) for e in (1, 2, 3)) + "\n")
    (d / "fold3_epochs.jsonl").write_text(json.dumps({"epoch": 1, "train_loss": 9, "val_loss": 9, "train_acc": 0, "val_acc": 0}) + "\n")
    c = load_curves(tmp_path, "crnn")
    assert c["train_loss"].shape == (2, 3)                               # the interrupted fold-3 log is dropped
    s = summarize(c)
    assert s["train_loss"]["mean"] == pytest.approx([0.9 + 0.1, 0.8 + 0.1, 0.7 + 0.1])
    assert s["train_loss"]["hi"] - s["train_loss"]["lo"] == pytest.approx([0.2, 0.2, 0.2])
    assert load_curves(tmp_path, "nothing") == {}
