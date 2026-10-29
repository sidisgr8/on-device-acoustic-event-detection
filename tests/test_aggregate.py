import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis import aggregate as A
from src.analysis.errors import analysis, normalise_rows, pooled_per_class_f1, summed_confusion, top_confusions
from conftest import CLASSES, write_fold


def test_refuses_to_mix_protocol_versions(tmp_path):
    write_fold(tmp_path, "cnn_baseline", 1)
    write_fold(tmp_path, "cnn_baseline", 2, protocol=1)
    with pytest.raises(A.ProtocolMixError, match="non-v2"):
        A.load_experiment(tmp_path, "cnn_baseline")
    with pytest.raises(A.ProtocolMixError):
        A.master_table(tmp_path)


def test_refuses_mixed_config_hashes(tmp_path):
    write_fold(tmp_path, "crnn", 1, cfg_hash="a")
    write_fold(tmp_path, "crnn", 2, cfg_hash="b")
    with pytest.raises(A.ProtocolMixError, match="different configs"):
        A.load_experiment(tmp_path, "crnn")


def test_master_table_values_and_fold_subset_label(tmp_path):
    for k, acc in zip((1, 5, 9), (0.80, 0.84, 0.88)):
        write_fold(tmp_path, "ast", k, acc=acc, n_params=85_625_098)
    for k in range(1, 11):
        write_fold(tmp_path, "cnn_baseline", k, acc=0.70)
    df = A.master_table(tmp_path, tmp_path / "ck")
    ast = df[df["_experiment"] == "ast"].iloc[0]
    assert ast["Folds"] == "3/10 (folds 1,5,9)"
    assert ast["Accuracy (%)"] == "84.00 ± 4.00"              # sample std of 80/84/88
    assert ast["Params (M)"] == "85.625"
    cnn = df[df["_experiment"] == "cnn_baseline"].iloc[0]
    assert cnn["Folds"] == "10/10" and cnn["Accuracy (%)"] == "70.00 ± 0.00"


def test_single_fold_has_no_std(tmp_path):
    write_fold(tmp_path, "crnn", 1, acc=0.7)
    assert "(n=1)" in A.master_table(tmp_path, tmp_path / "ck").iloc[0]["Accuracy (%)"]


def test_padded_vs_full_is_pooled(tmp_path):
    write_fold(tmp_path, "crnn", 1, acc=0.8)
    write_fold(tmp_path, "crnn", 2, acc=0.6)
    r = A.padded_vs_full_table(tmp_path).iloc[0]
    assert r["Padded clips (n)"] == 200 and r["Full-length clips (n)"] == 1400
    assert r["Full acc (%)"] == "70.00" and r["Padded acc (%)"] == "60.00"


def put(root, sub, exp, name, fold, **rec):
    d = Path(root) / sub / exp / name
    d.mkdir(parents=True, exist_ok=True)
    (d / f"fold{fold}.json").write_text(json.dumps({"experiment": exp, "fold": fold, "protocol_version": 2, **rec}))


def test_quant_table_uses_test_split_only_and_paired_delta(tmp_path):
    for k in (1, 2):
        write_fold(tmp_path, "cnn_baseline", k)
        put(tmp_path, "quant", "cnn_baseline", "fp32_torch", k, variant="fp32_torch", split="test",
            accuracy=0.70, macro_f1=0.68, size_bytes=1_600_000, size_ratio_vs_fp32=1.0, fp32_size_bytes=1_600_000)
        put(tmp_path, "quant", "cnn_baseline", "onnx_int8", k, variant="onnx_int8", split="test",
            accuracy=0.69, macro_f1=0.67, size_bytes=400_000, size_ratio_vs_fp32=0.25, fp32_size_bytes=1_600_000)
    put(tmp_path, "quant", "cnn_baseline", "onnx_int8", 3, variant="onnx_int8", split="val",    # validation dry run
        accuracy=0.10, macro_f1=0.1, size_bytes=1, size_ratio_vs_fp32=0.0, fp32_size_bytes=1)
    t = A.quant_table(tmp_path)
    row = t[t.Variant == "onnx_int8"].iloc[0]
    assert row["Folds"] == "2/10 (folds 1,2)"                 # the validation file is excluded
    assert row["Δ acc vs fp32 (pts)"].startswith("-1.00")
    assert row["Size vs fp32"] == "0.25×"


def test_quant_failure_is_reported_not_averaged(tmp_path):
    put(tmp_path, "quant", "crnn", "onnx_int8", 1, variant="onnx_int8", split="test", error="boom")
    assert A.quant_table(tmp_path).iloc[0]["Accuracy (%)"] == "failed"


def test_prune_table_excludes_dry_runs(tmp_path):
    write_fold(tmp_path, "cnn_baseline", 1, acc=0.70)
    common = dict(sparsity=0.5, nonzero_params=200_000, size_gzip_bytes=800_000,
                  size_gzip_bytes_unpruned_baseline=1_460_000, macro_f1=0.6)
    put(tmp_path, "prune", "cnn_baseline", "s50", 1, accuracy=0.68, split_evaluated="test", **common)
    put(tmp_path, "prune", "cnn_baseline", "s50", 5, accuracy=0.01, dry_run=True, split_evaluated="val", **common)
    r = A.prune_table(tmp_path).iloc[0]
    assert r["Folds"] == "1/10 (folds 1)" and r["Sparsity"] == "50%"
    assert r["Δ acc vs unpruned (pts)"].startswith("-2.00")


def test_compress_files_with_other_protocol_raise(tmp_path):
    put(tmp_path, "quant", "crnn", "onnx_int8", 1, variant="x", split="test")
    p = tmp_path / "quant" / "crnn" / "onnx_int8" / "fold1.json"
    d = json.loads(p.read_text())
    d["protocol_version"] = 1
    p.write_text(json.dumps(d))
    with pytest.raises(A.ProtocolMixError):
        A.quant_table(tmp_path)


def test_error_analysis_functions():
    cm = np.array([[8, 2, 0], [1, 7, 2], [0, 0, 10]])
    assert np.allclose(normalise_rows(cm).sum(axis=1), 1)
    top = top_confusions(cm, ["a", "b", "c"], n=2)
    assert (top.iloc[0].true, top.iloc[0].predicted, top.iloc[0]["count"]) in (("a", "b", 2), ("b", "c", 2))
    f1 = pooled_per_class_f1(cm)
    assert f1.shape == (3,) and f1[2] == pytest.approx(2 * (10 / 12) * 1 / (10 / 12 + 1))
    folds = [{"confusion_matrix": cm.tolist()}, {"confusion_matrix": cm.tolist()}]
    assert (summed_confusion(folds) == 2 * cm).all()


def test_analysis_from_results(tmp_path):
    for e in ("cnn_baseline", "crnn"):
        for k in (1, 2):
            write_fold(tmp_path, e, k)
    an = analysis(tmp_path)
    assert list(an) == ["cnn_baseline", "crnn"] and an["crnn"]["cm"].shape == (10, 10)
    assert an["crnn"]["f1_mean"].shape == (10,) and len(an["crnn"]["top"]) == 5


def put_complexity(root, macs):
    (Path(root)).mkdir(parents=True, exist_ok=True)
    (Path(root) / "complexity.json").write_text(json.dumps({"models": {
        e: {"macs_total": m, "params": 1} for e, m in macs.items()}}))


def test_master_table_has_params_size_macs_side_by_side(tmp_path):
    write_fold(tmp_path, "cnn_baseline", 1, n_params=390_890)
    write_fold(tmp_path, "ast", 1, n_params=85_625_098)
    df = A.master_table(tmp_path, tmp_path / "ck")
    cols = [c for c in df.columns if not c.startswith("_")]
    assert cols.index("Params (M)") < cols.index("fp32 size (MB)") < cols.index("MACs (M)") \
        < cols.index("CPU latency 1T median (ms)") < cols.index("CPU latency 4T median (ms)")
    assert (df["MACs (M)"] == "-").all()                                  # no complexity file yet
    put_complexity(tmp_path, {"cnn_baseline": 361_300_000, "ast": 44_082_900_000})
    df = A.master_table(tmp_path, tmp_path / "ck")
    assert df[df._experiment == "cnn_baseline"].iloc[0]["MACs (M)"] == "361.3"
    assert df[df._experiment == "ast"].iloc[0]["MACs (M)"] == "44,082.9"


def test_top_confusions_table(tmp_path):
    from src.analysis.errors import top_confusions_table
    for e in ("cnn_baseline", "ast"):
        for k in (1, 2):
            write_fold(tmp_path, e, k)
    t = top_confusions_table(tmp_path, n=3)
    assert len(t) == 6 and set(t.Model) == {"CNN baseline", "AST"}
    assert list(t[t.Model == "AST"]["Rank"]) == [1, 2, 3]
    assert (t["Clips"].values[:2] >= t["Clips"].values[1:3]).all()          # sorted by count within a model


def test_prune_table_has_matched_fold_baseline_and_labelled_control(tmp_path):
    for k, acc in ((1, 0.70), (5, 0.80), (9, 0.90), (2, 0.10)):          # fold 2 is not a pruning fold
        write_fold(tmp_path, "cnn_baseline", k, acc=acc)
    common = dict(nonzero_params=300_000, size_gzip_bytes=1_000_000, size_gzip_bytes_unpruned_baseline=1_460_000, macro_f1=0.6)
    for sp, name, delta in ((0.0, "s0", 0.01), (0.5, "s50", -0.05)):
        for k, base in ((1, 0.70), (5, 0.80), (9, 0.90)):
            put(tmp_path, "prune", "cnn_baseline", name, k, accuracy=base + delta, split_evaluated="test", sparsity=sp, **common)
    df = A.prune_table(tmp_path)
    ctrl, pr = df.iloc[0], df.iloc[1]
    assert ctrl["Sparsity"] == "0% (fine-tune control)" and pr["Sparsity"] == "50%"
    assert ctrl["Unpruned acc, same folds (%)"] == "80.00 ± 10.00"       # folds 1,5,9 only, not fold 2
    assert ctrl["Δ acc vs unpruned (pts)"].startswith("1.00") and pr["Δ acc vs unpruned (pts)"].startswith("-5.00")


def test_prune_table_extended_recipe_is_additional_labelled_row(tmp_path):
    for k, acc in ((1, 0.70), (5, 0.80), (9, 0.90)):
        write_fold(tmp_path, "cnn_baseline", k, acc=acc)
    common = dict(nonzero_params=300_000, size_gzip_bytes=1_000_000, size_gzip_bytes_unpruned_baseline=1_460_000, macro_f1=0.6,
                  split_evaluated="test", sparsity=0.5)
    for k, base in ((1, 0.70), (5, 0.80), (9, 0.90)):
        put(tmp_path, "prune", "cnn_baseline", "s50", k, accuracy=base - 0.2, **common)
        put(tmp_path, "prune", "cnn_baseline", "s50_ext", k, accuracy=base - 0.01, recipe="extended", **common)
    df = A.prune_table(tmp_path)
    assert len(df) == 2 and df.iloc[0]["Sparsity"] == "50%"                      # main row kept, unchanged
    assert df.iloc[1]["Sparsity"].startswith("50% [extended fine-tune recipe (lr and epochs chosen on validation data of the ImageNet")
    assert df.iloc[0]["Δ acc vs unpruned (pts)"].startswith("-20.00") and df.iloc[1]["Δ acc vs unpruned (pts)"].startswith("-1.00")
