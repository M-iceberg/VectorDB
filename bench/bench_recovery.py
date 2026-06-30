"""
Recovery time benchmark: WAL replay latency at different WAL sizes.

Inserts N vectors without calling checkpoint, closes the engine (WAL only,
no snapshot), then measures how long Engine reopen (pure C++ WAL replay)
takes. Insert is via Python SDK (slow); recovery is pure C++ (fast) — this
separates the two concerns.

Default N values produce ~2.7 MB / ~10.8 MB / ~27 MB WAL at dim=128.
Use --ci for smaller values suitable for CI (insert < 30s total).

Usage:
    python bench/bench_recovery.py [--dim 128] [--out bench_results/recovery]
    python bench/bench_recovery.py --ci
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import vectordb

# (label, n_vectors)
LEVELS_DEFAULT = [("small", 5_000), ("medium", 20_000), ("large", 50_000)]
LEVELS_CI      = [("small", 1_000), ("medium",  3_000), ("large",  8_000)]


def wal_mb(db_dir: str, collection: str) -> float:
    path = os.path.join(db_dir, collection, "wal.log")
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0


def run_level(label: str, n: int, dim: int, base_dir: str) -> dict:
    db_dir = os.path.join(base_dir, f"db_{label}")
    shutil.rmtree(db_dir, ignore_errors=True)

    rng = np.random.default_rng(42)

    # --- Setup: insert N vectors, no checkpoint ---
    db = vectordb.open(db_dir)
    db.create_collection("col", dimension=dim, metric="l2")

    BATCH = 500
    t_ins = time.perf_counter()
    for start in range(0, n, BATCH):
        batch = min(BATCH, n - start)
        vecs  = rng.random((batch, dim), dtype=np.float32)
        ids   = list(range(start, start + batch))
        db.insert("col", ids=ids, vectors=vecs)
    insert_sec = time.perf_counter() - t_ins

    size_mb = wal_mb(db_dir, "col")

    # Close without checkpoint — only WAL on disk.
    del db

    # --- Measure: C++ WAL replay on reopen ---
    t0 = time.perf_counter()
    db2 = vectordb.open(db_dir)
    recovery_ms = (time.perf_counter() - t0) * 1000

    # Verify recovery correctness.
    q = rng.random(dim, dtype=np.float32)
    results = db2.search("col", query=q, top_k=min(n, 10))
    assert len(results) > 0, f"WAL replay produced no results for n={n}"
    del db2

    shutil.rmtree(db_dir)

    return dict(label=label, n=n, dim=dim,
                wal_mb=size_mb, insert_sec=insert_sec,
                recovery_ms=recovery_ms,
                replay_kv_per_s=n / (recovery_ms / 1000) / 1000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--out", default="bench_results/recovery")
    parser.add_argument("--ci",  action="store_true",
                        help="Use small N values for CI (total insert < 30s)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    levels = LEVELS_CI if args.ci else LEVELS_DEFAULT

    # WAL record size estimate for informational header.
    # Header: 4+4+1+8=17B  Payload: 4(id)+2(uid_len)+5(uid)+dim*4
    bytes_per_rec = 17 + 4 + 2 + 5 + args.dim * 4
    print(f"Recovery benchmark  dim={args.dim}  ~{bytes_per_rec} B/WAL-record")
    print()
    print(f"{'Label':<8} {'N':>8} {'WAL MB':>8} {'Insert s':>10} "
          f"{'Recovery ms':>13} {'Replay Kv/s':>13}")
    print("-" * 67)

    rows = []
    for label, n in levels:
        r = run_level(label, n, args.dim, args.out)
        rows.append(r)
        print(f"{r['label']:<8} {r['n']:>8,} {r['wal_mb']:>8.2f} "
              f"{r['insert_sec']:>10.1f} {r['recovery_ms']:>13.1f} "
              f"{r['replay_kv_per_s']:>13.1f}")

    print()
    print("Recovery time = C++ WAL replay only (Engine.__init__ in C++).")
    print("Insert time = Python SDK overhead; not part of recovery path.")

    summary = dict(dim=args.dim, ci=args.ci, levels=rows)
    path = os.path.join(args.out, "results.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
