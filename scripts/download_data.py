"""Download and verify UrbanSound8K from Zenodo (about 6 GB), then extract to data/.

    python scripts/download_data.py

Uses curl (resumes automatically with -C -, and uses the system certificate store)
and then checks the MD5 and the file count. NOTE: the URL and MD5 were taken from the
Zenodo record page (record 1203745); the script fails loudly if either is wrong.
"""
import hashlib
import subprocess
import sys
import tarfile
from pathlib import Path

URL = "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz?download=1"
MD5 = "9aa69802bbf37fb986f71ec1483a196e"
DATA = Path("data")
ARCHIVE = DATA / "UrbanSound8K.tar.gz"


def md5(path):
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    DATA.mkdir(exist_ok=True)
    subprocess.run(["curl", "-L", "-C", "-", "--retry", "5", "--retry-delay", "5", "--fail",
                    "-o", str(ARCHIVE), URL], check=True)   # nothing runs after curl except the checks below
    got = md5(ARCHIVE)
    if got != MD5:
        sys.exit(f"MD5 mismatch: got {got}, expected {MD5}. Delete {ARCHIVE} and retry.")
    print("MD5 OK, extracting ...")
    with tarfile.open(ARCHIVE) as tar:
        tar.extractall(DATA)
    n = len(list((DATA / "UrbanSound8K" / "audio").glob("fold*/*.wav")))
    if n != 8732:
        sys.exit(f"expected 8732 WAV files, found {n}")
    print(f"done: {n} WAV files in {DATA/'UrbanSound8K'}")


if __name__ == "__main__":
    main()
