"""Build the shared caches (waveforms -> log-mel -> AST fbank).

    python scripts/build_cache.py [--stages waveforms logmel ast]
"""
import argparse
import time

from src.data import cache as C



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(C.DEFAULT_RAW))
    ap.add_argument("--cache", default=str(C.DEFAULT_CACHE))
    ap.add_argument("--stages", nargs="+", default=["waveforms", "logmel", "ast"],
                    choices=["waveforms", "logmel", "ast"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    for stage in a.stages:
        t = time.time()
        print(f"[{stage}]", flush=True)
        if stage == "waveforms":
            C.build_waveforms(a.raw, a.cache, a.workers)
        elif stage == "logmel":
            C.build_logmel(a.cache, a.workers)
        else:
            C.build_ast(a.cache, min(a.workers, 4))
        print(f"[{stage}] done in {time.time() - t:.0f}s", flush=True)
    C.write_meta(a.cache)


# Required: macOS multiprocessing re-imports this file in every worker.
if __name__ == "__main__":
    main()
