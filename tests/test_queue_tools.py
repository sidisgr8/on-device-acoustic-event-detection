import json
import subprocess
import sys
from pathlib import Path

from src.train.amp_probe import run_probe, select_mode

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "commit_results.py"


def modes(**kw):
    base = {"fp32": dict(eligible=True, median_step_s=1.0)}
    base.update({k: dict(eligible=v[0], median_step_s=v[1]) for k, v in kw.items()})
    return base


def test_select_mode_rule():
    assert select_mode(modes(fp16=(True, 0.6), bf16=(True, 0.5))) == "bf16"
    assert select_mode(modes(fp16=(True, 0.6), bf16=(False, 0.1))) == "fp16"   # fast but not eligible
    assert select_mode(modes(fp16=(False, 0.6), bf16=(False, 0.5))) == "fp32"
    assert select_mode(modes(fp16=(True, 1.5))) == "fp32"                      # slower than fp32


def test_probe_falls_back_to_fp32_on_failure(tmp_path):
    class Broken:
        fold = []
        def ast(self):
            raise RuntimeError("boom")
    msgs = []
    dec = tmp_path / "DECISIONS.md"
    dec.write_text("| D-27 | x |\n")
    r = run_probe({"model": {"name": "ast"}, "train": {}}, Broken(), __import__("torch").device("cpu"),
                  tmp_path / "probe.json", decisions_path=dec, log=msgs.append)
    assert r["chosen"] == "fp32" and "failed" in r["reason"]
    assert json.loads((tmp_path / "probe.json").read_text())["chosen"] == "fp32"
    assert "D-28" in dec.read_text() and "fp32" in dec.read_text().splitlines()[-1]


def test_commit_helper_commits_only_finished_v2_folds(tmp_path):
    def sh(*a):
        return subprocess.run(a, cwd=tmp_path, capture_output=True, text=True, check=True).stdout
    sh("git", "init", "-q")
    sh("git", "config", "user.email", "t@t")
    sh("git", "config", "user.name", "t")
    (tmp_path / "README").write_text("x")
    sh("git", "add", "README")
    sh("git", "commit", "-qm", "init")
    r = tmp_path / "results"
    (r / "exp").mkdir(parents=True)
    (r / "superseded_protocol_v1" / "exp").mkdir(parents=True)
    # finished v2 fold 1
    (r / "exp" / "fold1.json").write_text(json.dumps({"fold": 1, "protocol_version": 2}))
    (r / "exp" / "fold1_preds.csv").write_text("a")
    (r / "exp" / "fold1_epochs.jsonl").write_text("{}\n")
    # partial fold 2 (running): only an epoch log
    (r / "exp" / "fold2_epochs.jsonl").write_text("{}\n")
    # v1 fold 3 (no protocol_version) and a superseded dir
    (r / "exp" / "fold3.json").write_text(json.dumps({"fold": 3}))
    (r / "superseded_protocol_v1" / "exp" / "fold1.json").write_text(json.dumps({"fold": 1, "protocol_version": 2}))
    out = subprocess.run([sys.executable, str(SCRIPT)], cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    committed = sh("git", "show", "--name-only", "--format=", "HEAD").split()
    assert sorted(committed) == ["results/exp/fold1.json", "results/exp/fold1_epochs.jsonl",
                                 "results/exp/fold1_preds.csv"]
    untracked = sh("git", "status", "--porcelain")
    assert "fold2_epochs.jsonl" in untracked and "fold3.json" in untracked


def test_commit_helper_handles_post_results_and_skips_dry_runs_and_validation(tmp_path):
    def sh(*a):
        return subprocess.run(a, cwd=tmp_path, capture_output=True, text=True, check=True).stdout
    sh("git", "init", "-q"); sh("git", "config", "user.email", "t@t"); sh("git", "config", "user.name", "t")
    (tmp_path / "README").write_text("x"); sh("git", "add", "README"); sh("git", "commit", "-qm", "init")
    r = tmp_path / "results"
    def w(rel, obj):
        p = r / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj) if not isinstance(obj, str) else obj)
    w("quant/m/onnx_int8/fold1.json", {"protocol_version": 2, "split": "test", "accuracy": 0.7})
    w("quant/m/onnx_int8/fold2.json", {"protocol_version": 2, "split": "val", "accuracy": 0.7})       # validation: skipped
    w("prune/m/s30/fold1.json", {"protocol_version": 2, "accuracy": 0.7})
    w("prune/m/s30/fold5.json", {"protocol_version": 2, "dry_run": True, "val_accuracy": 0.7})        # dry run: skipped
    w("latency/latency.json", {"meta": {}, "forward": []})
    w("tables/tables.md", "# t"); w("figures/a.png", "png")
    out = subprocess.run([sys.executable, str(SCRIPT)], cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    log = sh("git", "log", "--name-only", "--format=%s").splitlines()
    committed = {l for l in log if l.startswith("results/")}
    assert committed == {"results/quant/m/onnx_int8/fold1.json", "results/prune/m/s30/fold1.json",
                         "results/latency/latency.json", "results/tables/tables.md", "results/figures/a.png"}
    status = sh("git", "status", "--porcelain")
    assert "fold2.json" in status and "fold5.json" in status
