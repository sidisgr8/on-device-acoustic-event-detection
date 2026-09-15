"""Compute the log-mel cache once:  python scripts/build_cache.py"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import build_cache  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/UrbanSound8K")
ap.add_argument("--cache", default="cache")
ap.add_argument("--jobs", type=int, default=-1, help="-1 = all CPU cores")
a = ap.parse_args()
build_cache(a.root, a.cache, a.jobs)
