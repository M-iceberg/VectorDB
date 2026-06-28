"""
SIFT-1M benchmark for VortexDB.

Usage:
    python bench/bench_sift.py [--data data/sift-128-euclidean.hdf5]
                               [--n-base 1000000]
                               [--n-queries 10000]
                               [--out bench_results/sift]

Measures:
  - Insert throughput (vectors/sec)
  - For each ef_search: QPS and recall@1/10/100
  - Generates recall-vs-ef and QPS-vs-recall charts
"""
import argparse
import json
import os
import sys
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import vectordb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_hdf5(path):
    with h5py.File(path, "r") as f:
        train  = f["train"][:]    # (N, dim) float32
        test   = f["test"][:]     # (Q, dim) float32
        neighbors = f["neighbors"][:]  # (Q, 100) int32  — ground truth
    return train, test, neighbors


def compute_recall(result_ids, gt, k):
    """Fraction of queries where the true top-1 (gt[:,0]) is in top-k results."""
    gt_top1 = gt[:, 0]
    hits = sum(
        1 for i, res in enumerate(result_ids)
        if gt_top1[i] in res[:k]
    )
    return hits / len(result_ids)


def compute_recall_at_k(result_ids, gt, k):
    """Recall@k: fraction of ground-truth top-k found in returned top-k."""
    total, found = 0, 0
    for i, res in enumerate(result_ids):
        gt_k = set(gt[i, :k])
        total += len(gt_k)
        found += len(gt_k & set(res[:k]))
    return found / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      default="data/sift-128-euclidean.hdf5")
    parser.add_argument("--n-base",    type=int, default=1_000_000)
    parser.add_argument("--n-queries", type=int, default=10_000)
    parser.add_argument("--out",       default="bench_results/sift")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data_path = os.path.join(os.path.dirname(__file__), "..", args.data)

    print(f"Loading dataset from {data_path} ...")
    train, test, gt = load_hdf5(data_path)
    train  = train[:args.n_base].astype(np.float32)
    test   = test[:args.n_queries].astype(np.float32)
    gt     = gt[:args.n_queries]
    N, dim = train.shape
    Q      = len(test)
    print(f"  Base: {N:,} vectors, dim={dim}")
    print(f"  Queries: {Q:,}")

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------
    db_dir = os.path.join(args.out, "db")
    import shutil
    shutil.rmtree(db_dir, ignore_errors=True)

    print("\nInserting vectors ...")
    db = vectordb.open(db_dir)
    db.create_collection("sift", dimension=dim, metric="l2")

    BATCH = 10_000
    t0 = time.perf_counter()
    for start in range(0, N, BATCH):
        end  = min(start + BATCH, N)
        ids  = list(range(start, end))
        db.insert("sift", ids=ids, vectors=train[start:end])
        if (start // BATCH) % 10 == 0:
            elapsed = time.perf_counter() - t0
            vps = (start + BATCH) / max(elapsed, 1e-9)
            print(f"  {end:>7,}/{N:,}  {vps:,.0f} vec/s", end="\r", flush=True)

    insert_time = time.perf_counter() - t0
    insert_throughput = N / insert_time
    print(f"\n  Done: {insert_time:.1f}s  ({insert_throughput:,.0f} vec/s)")

    # ------------------------------------------------------------------
    # Search sweep over ef_search values
    # ------------------------------------------------------------------
    ef_values = [50, 100, 200, 400, 800]
    results_table = []

    print("\nSearching ...")
    for ef in ef_values:
        result_ids = []
        t0 = time.perf_counter()
        for q in test:
            res = db.search("sift", query=q, top_k=100, ef_search=ef)
            result_ids.append([r["id"] for r in res])
        elapsed = time.perf_counter() - t0

        qps       = Q / elapsed
        recall_1  = compute_recall(result_ids, gt, 1)
        recall_10 = compute_recall_at_k(result_ids, gt, 10)
        recall_100= compute_recall_at_k(result_ids, gt, 100)

        row = dict(ef_search=ef, qps=qps,
                   recall_1=recall_1, recall_10=recall_10, recall_100=recall_100)
        results_table.append(row)
        print(f"  ef={ef:>4}  QPS={qps:>7,.1f}  "
              f"R@1={recall_1:.3f}  R@10={recall_10:.3f}  R@100={recall_100:.3f}")

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    summary = dict(
        dataset="SIFT-1M",
        n_base=N, n_queries=Q, dim=dim,
        insert_throughput_vps=insert_throughput,
        insert_time_sec=insert_time,
        search=results_table,
    )
    json_path = os.path.join(args.out, "results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    efs       = [r["ef_search"]  for r in results_table]
    r1_vals   = [r["recall_1"]   for r in results_table]
    r10_vals  = [r["recall_10"]  for r in results_table]
    r100_vals = [r["recall_100"] for r in results_table]
    qps_vals  = [r["qps"]        for r in results_table]

    # Plot 1: recall vs ef_search
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(efs, r1_vals,   "o-", label="Recall@1")
    ax.plot(efs, r10_vals,  "s-", label="Recall@10")
    ax.plot(efs, r100_vals, "^-", label="Recall@100")
    ax.set_xlabel("ef_search")
    ax.set_ylabel("Recall")
    ax.set_title(f"VortexDB — SIFT-1M: Recall vs ef_search\n"
                 f"({N:,} vectors, dim={dim}, {Q:,} queries)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    recall_plot = os.path.join(args.out, "recall_vs_ef.png")
    fig.savefig(recall_plot, dpi=150)
    print(f"Chart saved to {recall_plot}")

    # Plot 2: QPS vs recall@1 (trade-off curve)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r1_vals, qps_vals, "o-", color="tab:red")
    for ef, r1, qps in zip(efs, r1_vals, qps_vals):
        ax.annotate(f"ef={ef}", (r1, qps),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.set_xlabel("Recall@1")
    ax.set_ylabel("QPS")
    ax.set_title(f"VortexDB — SIFT-1M: QPS vs Recall@1\n"
                 f"({N:,} vectors, dim={dim})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    qps_plot = os.path.join(args.out, "qps_vs_recall.png")
    fig.savefig(qps_plot, dpi=150)
    print(f"Chart saved to {qps_plot}")

    plt.close("all")
    print("\nDone.")
    return summary


if __name__ == "__main__":
    main()
