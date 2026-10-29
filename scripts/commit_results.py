"""Commit only FINISHED result files, never half-written ones. Commits only the listed paths.

Groups (each its own commit):
* training folds: results/<exp>/fold<k>.json with protocol_version == 2, plus its preds CSV and epoch log
  (complete once the JSON exists); also results/ast_amp_probe.json with the DECISIONS.md row it appended;
* compression: results/quant/**/fold*.json (split == test), results/prune/**/fold*.json (not dry runs),
  results/latency/latency.json;
* tables and figures: results/tables/*, results/figures/*.
Superseded v1 results, dry runs and validation-split quantization files are never committed.

    python scripts/commit_results.py [--dry-run]
"""
import argparse
import json
import subprocess
from pathlib import Path

RESULTS = Path("results")
SKIP_DIRS = {"eda", "quant", "prune", "latency", "tables", "figures"}


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def changed(paths):
    paths = [p for p in paths if p.exists()]
    if not paths:
        return []
    out = git("status", "--porcelain", "--untracked-files=all", "--", *map(str, paths))
    dirty = {line[3:].strip().strip('"') for line in out.splitlines()}
    return [p for p in paths if str(p) in dirty]


def read_json(p):
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return None  # missing, or being written


def training_group():
    files, groups = [], {}
    for jf in sorted(RESULTS.glob("*/fold*.json")):
        if jf.parent.name.startswith("superseded") or jf.parent.name in SKIP_DIRS:
            continue
        r = read_json(jf)
        if not r or r.get("protocol_version") != 2:
            continue
        stem = jf.with_suffix("")
        cand = changed([jf, Path(f"{stem}_preds.csv"), Path(f"{stem}_epochs.jsonl")])
        if cand:
            files += cand
            groups.setdefault(jf.parent.name, []).append(r["fold"])
    parts = [f"{e} folds {sorted(f)}" for e, f in groups.items()]
    probe = RESULTS / "ast_amp_probe.json"
    if changed([probe]):
        files += [probe, Path("docs/DECISIONS.md")]
        parts.append("AST precision probe")
    return ("Add results (protocol v2, real runs): " + "; ".join(parts), files) if files else None


def compression_group():
    files, parts = [], []
    for sub, label, ok in (("quant", "quantization", lambda r: r.get("split") == "test"),
                           ("prune", "pruning", lambda r: not r.get("dry_run") and "accuracy" in r)):
        sel = []
        for jf in sorted((RESULTS / sub).rglob("fold*.json")) if (RESULTS / sub).exists() else []:
            r = read_json(jf)
            if r and r.get("protocol_version") == 2 and ok(r):
                sel.append(jf)
        sel = changed(sel)
        if sel:
            files += sel
            parts.append(f"{label} ({len(sel)} files)")
    lat = RESULTS / "latency" / "latency.json"
    if read_json(lat) is not None and changed([lat]):
        files.append(lat)
        parts.append("latency")
    return ("Add compression results (real runs): " + "; ".join(parts), files) if files else None


def report_group():
    files = []
    for d in ("tables", "figures"):
        if (RESULTS / d).exists():
            files += changed(sorted(p for p in (RESULTS / d).iterdir() if p.is_file()))
    return ("Regenerate result tables and figures from result files", files) if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    groups = [g for g in (training_group(), compression_group(), report_group()) if g]
    if not groups:
        print("nothing to commit")
        return
    for msg, files in groups:
        print(msg)
        if a.dry_run:
            print("  " + "\n  ".join(map(str, files)))
            continue
        git("add", "--", *map(str, files))
        git("commit", "-m", msg, "--", *map(str, files))


if __name__ == "__main__":
    main()
