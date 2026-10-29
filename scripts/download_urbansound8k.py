"""Download UrbanSound8K from Zenodo (record 1203745) and verify its MD5.

Zenodo throttles each connection, so the archive is fetched as several
byte ranges in parallel. Every range resumes from what is already on disk,
so the script can be re-run after an interruption.

    python scripts/download_urbansound8k.py [--out data/raw] [--workers 16]
"""
import argparse
import hashlib
import json
import shutil
import ssl
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import certifi

RECORD_API = "https://zenodo.org/api/records/1203745"
CHUNK_READ = 1 << 20
# python.org builds ship no CA bundle, so pass certifi's explicitly.
CTX = ssl.create_default_context(cafile=certifi.where())


def get_file_info():
    with urllib.request.urlopen(RECORD_API, timeout=30, context=CTX) as r:
        rec = json.load(r)
    f = next(f for f in rec["files"] if f["key"] == "UrbanSound8K.tar.gz")
    algo, digest = f["checksum"].split(":")
    return f["links"]["self"], f["size"], algo, digest


def fetch_part(url, part_path, start, end, progress, retries=30):
    """Download bytes [start, end] into part_path, resuming if it exists."""
    want = end - start + 1
    for attempt in range(retries):
        have = part_path.stat().st_size if part_path.exists() else 0
        if have >= want:
            return
        req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=60, context=CTX) as r, open(part_path, "ab") as out:
                while True:
                    buf = r.read(CHUNK_READ)
                    if not buf:
                        break
                    out.write(buf)
                    progress.add(len(buf))
        except Exception as e:  # network hiccup: back off and resume
            time.sleep(min(2 * (attempt + 1), 20))
    raise RuntimeError(f"part {part_path.name} failed after {retries} attempts")


class Progress:
    def __init__(self, total, initial):
        self.total, self.done, self.lock, self.t0 = total, initial, threading.Lock(), time.time()
        self.initial = initial

    def add(self, n):
        with self.lock:
            self.done += n

    def report(self):
        rate = (self.done - self.initial) / max(time.time() - self.t0, 1e-6) / 1e6
        print(f"\r{self.done / 1e6:8.0f} / {self.total / 1e6:.0f} MB  {rate:5.2f} MB/s", end="", flush=True)


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while buf := f.read(CHUNK_READ):
            h.update(buf)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out)
    parts_dir = out / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    target = out / "UrbanSound8K.tar.gz"

    url, size, algo, digest = get_file_info()
    assert algo == "md5"
    if target.exists() and target.stat().st_size == size and md5_of(target) == digest:
        print("already downloaded and verified")
        return

    n = args.workers
    step = -(-size // n)
    ranges = [(i, i * step, min((i + 1) * step, size) - 1) for i in range(n)]
    initial = sum(p.stat().st_size for p in parts_dir.glob("part_*"))
    progress = Progress(size, initial)

    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            progress.report()
            time.sleep(5)

    threading.Thread(target=ticker, daemon=True).start()
    with ThreadPoolExecutor(n) as ex:
        futs = [ex.submit(fetch_part, url, parts_dir / f"part_{i:02d}", s, e, progress) for i, s, e in ranges]
        for f in futs:
            f.result()
    stop.set()
    progress.report()
    print("\nassembling")
    with open(target, "wb") as dst:
        for i, _, _ in ranges:
            with open(parts_dir / f"part_{i:02d}", "rb") as src:
                shutil.copyfileobj(src, dst, CHUNK_READ)
    got = md5_of(target)
    if got != digest:
        raise SystemExit(f"MD5 mismatch: expected {digest}, got {got}")
    shutil.rmtree(parts_dir)
    print(f"verified md5 {digest}")


if __name__ == "__main__":
    main()
