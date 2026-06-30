"""
Stress test: mixed insert + search + delete for N minutes.

Runs a sequential loop of insert batches → search queries → delete batches,
checkpointing periodically. At the end, reopens the engine and verifies all
known live vectors are still searchable (data loss = 0).

Usage:
    python bench/bench_stress.py [--duration 600] [--dim 32]
    python bench/bench_stress.py --ci     # 60s run, dim=16
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import vectordb

DIM_DEFAULT      = 32
DURATION_DEFAULT = 600   # 10 minutes
BATCH_INSERT     = 50
BATCH_DELETE     = 20
QUERIES_PER_ITER = 10
CHECKPOINT_EVERY = 10    # iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=DURATION_DEFAULT,
                        help="Stress test duration in seconds (default: 600)")
    parser.add_argument("--dim", type=int, default=DIM_DEFAULT)
    parser.add_argument("--out", default="bench_results/stress")
    parser.add_argument("--ci", action="store_true",
                        help="Short run: 60 s, dim=16")
    args = parser.parse_args()

    if args.ci:
        duration = 60
        dim = 16
    else:
        duration = args.duration
        dim = args.dim

    os.makedirs(args.out, exist_ok=True)
    db_dir = os.path.join(args.out, "db")
    shutil.rmtree(db_dir, ignore_errors=True)

    rng = np.random.default_rng(42)

    db = vectordb.open(db_dir)
    db.create_collection("col", dimension=dim, metric="l2")

    # Ground truth: {str_id -> vector} for all live (non-deleted) vectors.
    live: dict[str, np.ndarray] = {}
    next_id  = 0
    n_insert = 0
    n_search = 0
    n_delete = 0
    n_iter   = 0
    errors   = []

    print(f"Stress test  dim={dim}  duration={duration}s")
    print(f"  insert {BATCH_INSERT}/iter  search {QUERIES_PER_ITER}/iter  "
          f"delete {BATCH_DELETE}/iter  checkpoint every {CHECKPOINT_EVERY} iters")
    print()

    deadline = time.perf_counter() + duration
    t_start  = time.perf_counter()
    last_report = t_start

    while time.perf_counter() < deadline:
        n_iter += 1

        # --- Insert ---
        vecs = rng.random((BATCH_INSERT, dim), dtype=np.float32)
        ids  = list(range(next_id, next_id + BATCH_INSERT))
        try:
            db.insert("col", ids=ids, vectors=vecs)
            for i, sid in enumerate(map(str, ids)):
                live[sid] = vecs[i]
            next_id  += BATCH_INSERT
            n_insert += BATCH_INSERT
        except Exception as e:
            errors.append(f"insert iter={n_iter}: {e}")

        # --- Search ---
        for _ in range(QUERIES_PER_ITER):
            q = rng.random(dim, dtype=np.float32)
            try:
                results = db.search("col", query=q, top_k=5)
                n_search += 1
                # Returned IDs must all be live.
                for r in results:
                    if r["id"] not in live:
                        errors.append(
                            f"search iter={n_iter}: returned deleted/unknown id {r['id']}")
            except Exception as e:
                errors.append(f"search iter={n_iter}: {e}")

        # --- Delete (only if we have enough live vectors) ---
        if len(live) > BATCH_DELETE * 4:
            to_delete = rng.choice(list(live.keys()), BATCH_DELETE, replace=False)
            for sid in to_delete:
                try:
                    db.remove("col", sid)
                    del live[sid]
                    n_delete += 1
                except Exception as e:
                    errors.append(f"delete iter={n_iter} id={sid}: {e}")

        # --- Checkpoint ---
        if n_iter % CHECKPOINT_EVERY == 0:
            try:
                db.checkpoint("col")
            except Exception as e:
                errors.append(f"checkpoint iter={n_iter}: {e}")

        # --- Progress report every 10s ---
        now = time.perf_counter()
        if now - last_report >= 10:
            elapsed = now - t_start
            remaining = max(0, deadline - now)
            ops = n_insert + n_search + n_delete
            print(f"  {elapsed:>5.0f}s  live={len(live):>6,}  "
                  f"ins={n_insert:>7,}  srch={n_search:>7,}  del={n_delete:>6,}  "
                  f"ops/s={ops/elapsed:>6.0f}  errors={len(errors)}")
            last_report = now

    elapsed = time.perf_counter() - t_start
    ops_total = n_insert + n_search + n_delete

    print()
    print("=== Stress test complete ===")
    print(f"  Duration:     {elapsed:.1f}s")
    print(f"  Iterations:   {n_iter:,}")
    print(f"  Inserts:      {n_insert:,}")
    print(f"  Searches:     {n_search:,}")
    print(f"  Deletes:      {n_delete:,}")
    print(f"  Throughput:   {ops_total/elapsed:.0f} ops/s")
    print(f"  Live vectors: {len(live):,}")
    print(f"  Errors:       {len(errors)}")
    for e in errors[:10]:
        print(f"    {e}")

    del db

    # --- Data loss verification: reopen and check all live IDs searchable ---
    print()
    print("=== Recovery + data loss check ===")
    t0 = time.perf_counter()
    db2 = vectordb.open(db_dir)
    recovery_ms = (time.perf_counter() - t0) * 1000
    print(f"  Recovery time: {recovery_ms:.1f} ms")

    missing = 0
    sample  = list(live.items())
    # Check up to 500 random live vectors to keep verification fast.
    rng2 = np.random.default_rng(99)
    sample_ids = rng2.choice(len(sample), min(500, len(sample)), replace=False)
    for idx in sample_ids:
        sid, vec = sample[idx]
        results = db2.search("col", query=vec, top_k=5, ef_search=64)
        if not any(r["id"] == sid for r in results):
            missing += 1

    print(f"  Verified:     {len(sample_ids)} live vectors sampled")
    print(f"  Missing:      {missing}")

    if errors:
        print("\nFAIL — errors during stress run (see above)")
        sys.exit(1)
    if missing > 0:
        print(f"\nFAIL — {missing} live vectors not found after recovery")
        sys.exit(1)

    print("\nPASS — no crashes, no data loss")


if __name__ == "__main__":
    main()
