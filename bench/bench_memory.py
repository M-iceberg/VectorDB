"""
Memory profiling for VortexDB HNSW index.

Measures peak RSS at different collection sizes and computes
bytes-per-vector overhead beyond raw vector storage.

Usage:
    /usr/bin/time -l python bench/bench_memory.py          # macOS
    /usr/bin/time -v python bench/bench_memory.py          # Linux

Or run directly for per-checkpoint measurement:
    python bench/bench_memory.py [--dim 128] [--n 200000] [--out bench_results/memory]
"""
import argparse
import json
import os
import sys
import resource
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import vectordb


def rss_mb():
    """Current RSS in MB (works on macOS and Linux)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports in bytes, Linux in kilobytes
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    else:
        return usage.ru_maxrss / 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim",  type=int, default=128,
                        help="Vector dimension (default: 128, SIFT-like)")
    parser.add_argument("--n",    type=int, default=500_000,
                        help="Total vectors to insert")
    parser.add_argument("--step", type=int, default=50_000,
                        help="Checkpoint interval for RSS measurement")
    parser.add_argument("--out",  default="bench_results/memory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    import shutil
    db_dir = os.path.join(args.out, "db")
    shutil.rmtree(db_dir, ignore_errors=True)

    rng = np.random.default_rng(42)
    raw_bytes_per_vec = args.dim * 4  # float32

    print(f"Memory profiling: dim={args.dim}, n={args.n:,}, step={args.step:,}")
    print(f"Raw vector storage: {raw_bytes_per_vec} bytes/vec")
    print()

    db = vectordb.open(db_dir)
    db.create_collection("mem", dimension=args.dim, metric="l2")

    baseline_rss = rss_mb()
    print(f"Baseline RSS (empty engine): {baseline_rss:.1f} MB")

    checkpoints = []
    BATCH = 5_000
    inserted = 0

    while inserted < args.n:
        batch_n = min(BATCH, args.n - inserted)
        vecs = rng.random((batch_n, args.dim), dtype=np.float32)
        ids  = list(range(inserted, inserted + batch_n))
        db.insert("mem", ids=ids, vectors=vecs)
        inserted += batch_n

        if inserted % args.step == 0 or inserted == args.n:
            rss = rss_mb()
            raw_mb = inserted * raw_bytes_per_vec / (1024 * 1024)
            overhead_mb = rss - baseline_rss - raw_mb
            overhead_bpv = overhead_mb * 1024 * 1024 / inserted if inserted > 0 else 0
            total_bpv = (rss - baseline_rss) * 1024 * 1024 / inserted if inserted > 0 else 0

            row = dict(
                n=inserted,
                rss_mb=rss,
                raw_mb=raw_mb,
                overhead_mb=overhead_mb,
                overhead_bytes_per_vec=overhead_bpv,
                total_bytes_per_vec=total_bpv,
            )
            checkpoints.append(row)
            print(f"  n={inserted:>7,}  RSS={rss:>7.1f}MB  "
                  f"raw={raw_mb:>6.1f}MB  overhead={overhead_mb:>6.1f}MB  "
                  f"({overhead_bpv:.0f} B/vec overhead, {total_bpv:.0f} B/vec total)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    final = checkpoints[-1]
    print()
    print("=== Summary ===")
    print(f"  Vectors:          {final['n']:,}")
    print(f"  Peak RSS:         {final['rss_mb']:.1f} MB")
    print(f"  Raw vector data:  {final['raw_mb']:.1f} MB  ({raw_bytes_per_vec} B/vec)")
    print(f"  HNSW overhead:    {final['overhead_mb']:.1f} MB  ({final['overhead_bytes_per_vec']:.0f} B/vec)")
    print(f"  Total:            {final['total_bytes_per_vec']:.0f} B/vec  "
          f"({final['total_bytes_per_vec'] / raw_bytes_per_vec:.1f}x raw)")

    summary = dict(
        dim=args.dim,
        n=args.n,
        raw_bytes_per_vec=raw_bytes_per_vec,
        peak_rss_mb=final["rss_mb"],
        hnsw_overhead_mb=final["overhead_mb"],
        hnsw_overhead_bytes_per_vec=final["overhead_bytes_per_vec"],
        total_bytes_per_vec=final["total_bytes_per_vec"],
        checkpoints=checkpoints,
    )
    json_path = os.path.join(args.out, "results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
