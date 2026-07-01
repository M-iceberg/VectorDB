"""
Head-to-head: VectorDB vs hnswlib on SIFT-1M.

Runs both on the same machine with identical M / ef_construction parameters,
sweeps ef_search, and plots the QPS vs Recall@10 trade-off curve — the same
metric used by ann-benchmarks.com.

Requirements:
    pip install hnswlib h5py matplotlib

Usage:
    python bench/bench_ann_compare.py [--data data/sift-128-euclidean.hdf5]
"""
import argparse
import os
import shutil
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import vectordb

try:
    import hnswlib
    HAS_HNSWLIB = True
except ImportError:
    HAS_HNSWLIB = False
    print("hnswlib not installed — only VectorDB results will be shown.")
    print("  pip install hnswlib\n")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ann-benchmarks primary metric: recall@10
# "fraction of true top-10 neighbors found in returned top-10"
def recall_at_10(returned: list[list], gt: np.ndarray) -> float:
    total = found = 0
    for i, res in enumerate(returned):
        true_top10 = set(gt[i, :10].tolist())
        total += len(true_top10)
        found += len(true_top10 & set(res[:10]))
    return found / total if total > 0 else 0.0


def load_sift(path, n_base, n_queries):
    with h5py.File(path, "r") as f:
        train = f["train"][:n_base].astype(np.float32)
        test  = f["test"][:n_queries].astype(np.float32)
        gt    = f["neighbors"][:n_queries]
    return train, test, gt


# ---------------------------------------------------------------------------
# VectorDB
# ---------------------------------------------------------------------------

def run_vectordb(train, test, gt, ef_values, M, ef_construction, out_dir):
    N, dim = train.shape
    db_dir = os.path.join(out_dir, "vdb")
    shutil.rmtree(db_dir, ignore_errors=True)

    db = vectordb.open(db_dir)
    db.create_collection("sift", dimension=dim, metric="l2")

    print("  [VectorDB] building index ...")
    t0 = time.perf_counter()
    BATCH = 10_000
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        db.insert("sift", ids=list(range(start, end)), vectors=train[start:end])
    build_sec = time.perf_counter() - t0
    print(f"  [VectorDB] build: {build_sec:.1f}s  ({N/build_sec:,.0f} vec/s)")

    rows = []
    Q = len(test)
    for ef in ef_values:
        returned = []
        t0 = time.perf_counter()
        for q in test:
            res = db.search("sift", query=q, top_k=10, ef_search=ef)
            returned.append([int(r["id"]) for r in res])
        elapsed = time.perf_counter() - t0
        qps = Q / elapsed
        rec = recall_at_10(returned, gt)
        rows.append((ef, qps, rec))
        print(f"  [VectorDB] ef={ef:>4}  QPS={qps:>7,.0f}  Recall@10={rec:.4f}")

    return build_sec, rows


# ---------------------------------------------------------------------------
# hnswlib
# ---------------------------------------------------------------------------

def run_hnswlib(train, test, gt, ef_values, M, ef_construction):
    if not HAS_HNSWLIB:
        return None, []

    N, dim = train.shape
    print("  [hnswlib]  building index ...")
    index = hnswlib.Index(space="l2", dim=dim)

    t0 = time.perf_counter()
    index.init_index(max_elements=N, ef_construction=ef_construction, M=M)
    index.add_items(train, list(range(N)))
    build_sec = time.perf_counter() - t0
    print(f"  [hnswlib]  build: {build_sec:.1f}s  ({N/build_sec:,.0f} vec/s)")

    rows = []
    Q = len(test)
    for ef in ef_values:
        index.set_ef(ef)
        t0 = time.perf_counter()
        labels, _ = index.knn_query(test, k=10)
        elapsed = time.perf_counter() - t0
        qps = Q / elapsed
        rec = recall_at_10(labels.tolist(), gt)
        rows.append((ef, qps, rec))
        print(f"  [hnswlib]  ef={ef:>4}  QPS={qps:>7,.0f}  Recall@10={rec:.4f}")

    return build_sec, rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",           default="data/sift-128-euclidean.hdf5")
    parser.add_argument("--n-base",         type=int, default=1_000_000)
    parser.add_argument("--n-queries",      type=int, default=10_000)
    parser.add_argument("--M",              type=int, default=16)
    parser.add_argument("--ef-construction",type=int, default=200)
    parser.add_argument("--out",            default="bench_results/ann_compare")
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(__file__), "..", args.data)
    os.makedirs(args.out, exist_ok=True)

    EF_VALUES = [50, 100, 200, 400, 800]

    print(f"Loading SIFT-1M from {data_path} ...")
    train, test, gt = load_sift(data_path, args.n_base, args.n_queries)
    print(f"  {len(train):,} base  {len(test):,} queries  dim={train.shape[1]}")
    print(f"  M={args.M}  ef_construction={args.ef_construction}\n")

    vdb_build, vdb_rows = run_vectordb(
        train, test, gt, EF_VALUES, args.M, args.ef_construction, args.out)

    print()
    hnsw_build, hnsw_rows = run_hnswlib(
        train, test, gt, EF_VALUES, args.M, args.ef_construction)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print()
    print(f"{'ef':>5}  {'VectorDB QPS':>14}  {'VectorDB R@10':>14}  "
          f"{'hnswlib QPS':>13}  {'hnswlib R@10':>13}  {'QPS ratio':>10}")
    print("-" * 80)
    for i, (ef, vqps, vrec) in enumerate(vdb_rows):
        if hnsw_rows and i < len(hnsw_rows):
            _, hqps, hrec = hnsw_rows[i]
            ratio = f"{hqps/vqps:.1f}x"
            print(f"{ef:>5}  {vqps:>14,.0f}  {vrec:>14.4f}  "
                  f"{hqps:>13,.0f}  {hrec:>13.4f}  {ratio:>10}")
        else:
            print(f"{ef:>5}  {vqps:>14,.0f}  {vrec:>14.4f}")

    print()
    print(f"Build time — VectorDB: {vdb_build:.1f}s", end="")
    if hnsw_rows:
        print(f"  hnswlib: {hnsw_build:.1f}s  (ratio: {vdb_build/hnsw_build:.1f}x slower)")
    else:
        print()

    # ------------------------------------------------------------------
    # Plot: QPS vs Recall@10 (ann-benchmarks style)
    # ------------------------------------------------------------------
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(8, 5))

        v_recs = [r for _, _, r in vdb_rows]
        v_qps  = [q for _, q, _ in vdb_rows]
        ax.plot(v_recs, v_qps, "o-", color="tab:blue",  label="VectorDB (Python SDK)")

        if hnsw_rows:
            h_recs = [r for _, _, r in hnsw_rows]
            h_qps  = [q for _, q, _ in hnsw_rows]
            ax.plot(h_recs, h_qps, "s-", color="tab:orange", label="hnswlib (Python)")

        for ef, qps, rec in vdb_rows:
            ax.annotate(f"ef={ef}", (rec, qps),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color="tab:blue")
        if hnsw_rows:
            for ef, qps, rec in hnsw_rows:
                ax.annotate(f"ef={ef}", (rec, qps),
                            textcoords="offset points", xytext=(4, -12),
                            fontsize=7, color="tab:orange")

        ax.set_xlabel("Recall@10")
        ax.set_ylabel("QPS")
        ax.set_title(f"SIFT-1M: QPS vs Recall@10  (M={args.M}, ef_construction={args.ef_construction})\n"
                     f"ann-benchmarks methodology — {len(train):,} vectors, dim={train.shape[1]}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = os.path.join(args.out, "qps_vs_recall10.png")
        fig.savefig(plot_path, dpi=150)
        print(f"Plot saved to {plot_path}")
        plt.close()


if __name__ == "__main__":
    main()
