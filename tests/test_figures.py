import json
from pathlib import Path

from src.analysis.figures import make_all
from conftest import write_fold

BASE = ["confusion_cnn_baseline.png", "confusion_mobilenetv3_small_scratch.png", "confusion_crnn.png",
        "confusion_ast.png", "confusion_grid.png", "top_confusions.png", "per_class_f1.png", "fold_accuracy.png",
        "padded_vs_full.png", "accuracy_vs_params.png", "accuracy_vs_size.png",
        "paired_differences.png", "paired_differences_macro_f1.png"]   # training_curves.png only with epoch logs
EXPS = ("cnn_baseline", "mobilenetv3_small_scratch", "crnn", "ast")


def synth(tmp_path):
    for i, e in enumerate(EXPS):
        for k in ((1, 5, 9) if e == "ast" else (1, 2, 3)):
            write_fold(tmp_path / "res", e, k, acc=0.6 + 0.05 * i, n_params=10 ** (5 + i))


def test_all_figures_are_written_and_reproducible(tmp_path):
    synth(tmp_path)
    files = make_all(tmp_path / "res", tmp_path / "figs")
    assert sorted(p.name for p in files) == sorted(BASE)        # no complexity/latency data -> no companion figures
    assert all(p.stat().st_size > 5_000 for p in files)
    first = {p.name: p.read_bytes() for p in files}
    make_all(tmp_path / "res", tmp_path / "figs2")
    assert all((tmp_path / "figs2" / n).read_bytes() == b for n, b in first.items())   # deterministic
    assert make_all(tmp_path / "empty", tmp_path / "x") == []


def test_accuracy_vs_macs_figure_appears_when_complexity_exists(tmp_path):
    synth(tmp_path)
    (tmp_path / "res" / "complexity.json").write_text(json.dumps({"models": {
        "cnn_baseline": {"macs_total": 361_300_000}, "mobilenetv3_small_scratch": {"macs_total": 28_600_000},
        "crnn": {"macs_total": 266_300_000}, "ast": {"macs_total": 44_082_900_000}}}))
    files = make_all(tmp_path / "res", tmp_path / "figs")
    names = {p.name for p in files}
    assert "accuracy_vs_macs.png" in names and "accuracy_vs_params.png" in names
    assert "accuracy_vs_latency.png" not in names
    assert (tmp_path / "figs" / "accuracy_vs_macs.png").stat().st_size > 5_000


def test_accuracy_vs_latency_figure_appears_when_latency_exists(tmp_path):
    synth(tmp_path)
    (tmp_path / "res" / "latency").mkdir()
    (tmp_path / "res" / "latency" / "latency.json").write_text(json.dumps({"meta": {}, "feature_extraction": [], "forward": [
        {"experiment": e, "variant": "fp32_torch", "threads": 1, "median_ms": 5.0 * (i + 1), "p95_ms": 6.0,
         "size_bytes": 1000, "stage": "forward"} for i, e in enumerate(EXPS)]}))
    names = {p.name for p in make_all(tmp_path / "res", tmp_path / "figs")}
    assert "accuracy_vs_latency.png" in names


def test_training_curves_figure_needs_epoch_logs(tmp_path):
    synth(tmp_path)
    for e in EXPS:
        for k in (1, 2):
            (tmp_path / "res" / e / f"fold{k}_epochs.jsonl").write_text("\n".join(json.dumps(
                {"epoch": i, "train_loss": 1.0 / i, "val_loss": 1.2 / i, "train_acc": 0.5, "val_acc": 0.4}) for i in (1, 2, 3)) + "\n")
    names = {p.name for p in make_all(tmp_path / "res", tmp_path / "figs")}
    assert "training_curves.png" in names and set(BASE) <= names
