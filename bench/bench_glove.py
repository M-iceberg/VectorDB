"""
GloVe-1.2M benchmark — VectorDB vs hnswlib vs faiss.

Usage:
    python bench/bench_glove.py [--data data/glove-200-angular.hdf5]

Measures:
  - Build throughput (vec/s)
  - For each ef_search: QPS and Recall@10
  - Head-to-head comparison table vs hnswlib and faiss
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
    print("hnswlib not installed — skipping hnswlib.")

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("faiss not installed — skipping faiss.  pip install faiss-cpu")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_hdf5(path, n_base, n_queries):
    with h5py.File(path, "r") as f:
        train     = f["train"][:n_base].astype(np.float32)
        test      = f["test"][:n_queries].astype(np.float32)
        neighbors = f["neighbors"][:n_queries]
    return train, test, neighbors


def recall_at_10(returned, gt):
    total = found = 0
    for i, res in enumerate(returned):
        true_top10 = set(int(x) for x in gt[i, :10])
        total += len(true_top10)
        found += len(true_top10 & set(res[:10]))
    return found / total if total > 0 else 0.0


def run_vectordb(train, test, gt, ef_values, out_dir):
    N, dim = train.shape
    db_dir = os.path.join(out_dir, "db")
    shutil.rmtree(db_dir, ignore_errors=True)

    db = vectordb.open(db_dir)
    db.create_collection("glove", dimension=dim, metric="cosine")

    print("  [VectorDB] building index ...")
    BATCH = 10_000
    t0 = time.perf_counter()
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        db.insert("glove", ids=list(range(start, end)), vectors=train[start:end])
    build_sec = time.perf_counter() - t0
    print(f"  [VectorDB] build: {build_sec:.1f}s  ({N/build_sec:,.0f} vec/s)")

    rows = []
    Q = len(test)
    for ef in ef_values:
        t0 = time.perf_counter()
        batch = db.search_batch("glove", queries=test, top_k=10, ef_search=ef)
        elapsed = time.perf_counter() - t0
        returned = [[int(r["id"]) for r in row] for row in batch]
        qps = Q / elapsed
        rec = recall_at_10(returned, gt)
        rows.append((ef, qps, rec))
        print(f"  [VectorDB] ef={ef:>4}  QPS={qps:>7,.0f}  Recall@10={rec:.4f}")

    return build_sec, rows


def run_hnswlib(train, test, gt, ef_values):
    if not HAS_HNSWLIB:
        return None, []

    N, dim = train.shape
    print("  [hnswlib]  building index ...")
    index = hnswlib.Index(space="cosine", dim=dim)

    t0 = time.perf_counter()
    index.init_index(max_elements=N, ef_construction=200, M=16)
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


def run_faiss(train, test, gt, ef_values):
    if not HAS_FAISS:
        return None, []

    N, dim = train.shape
    # faiss has no native cosine index; normalize vectors to use inner product
    # as cosine proxy (same approach hnswlib uses internally).
    norms = np.linalg.norm(train, axis=1, keepdims=True)
    train_n = train / np.where(norms == 0, 1, norms)
    norms_q = np.linalg.norm(test, axis=1, keepdims=True)
    test_n  = test / np.where(norms_q == 0, 1, norms_q)

    print("  [faiss]    building index ...")
    index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200

    t0 = time.perf_counter()
    index.add(train_n)
    build_sec = time.perf_counter() - t0
    print(f"  [faiss]    build: {build_sec:.1f}s  ({N/build_sec:,.0f} vec/s)")

    rows = []
    Q = len(test)
    for ef in ef_values:
        index.hnsw.efSearch = ef
        t0 = time.perf_counter()
        _, labels = index.search(test_n, 10)
        elapsed = time.perf_counter() - t0
        qps = Q / elapsed
        rec = recall_at_10(labels.tolist(), gt)
        rows.append((ef, qps, rec))
        print(f"  [faiss]    ef={ef:>4}  QPS={qps:>7,.0f}  Recall@10={rec:.4f}")

    return build_sec, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      default="data/glove-200-angular.hdf5")
    parser.add_argument("--n-base",    type=int, default=1_183_514)
    parser.add_argument("--n-queries", type=int, default=10_000)
    parser.add_argument("--out",       default="bench_results/glove")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data_path = os.path.join(os.path.dirname(__file__), "..", args.data)

    EF_VALUES = [50, 100, 200, 400, 800]

    print(f"Loading GloVe from {data_path} ...")
    train, test, gt = load_hdf5(data_path, args.n_base, args.n_queries)
    N, dim = train.shape
    print(f"  {N:,} base  {len(test):,} queries  dim={dim}  metric=cosine")
    print(f"  M=16  ef_construction=200\n")

    vdb_build, vdb_rows = run_vectordb(train, test, gt, EF_VALUES, args.out)

    print()
    hnsw_build, hnsw_rows = run_hnswlib(train, test, gt, EF_VALUES)

    print()
    faiss_build, faiss_rows = run_faiss(train, test, gt, EF_VALUES)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print()
    header = f"{'ef':>5}  {'VectorDB':>10} {'R@10':>6}  {'hnswlib':>10} {'R@10':>6}  {'faiss':>10} {'R@10':>6}"
    print(header)
    print("-" * len(header))
    for i, (ef, vqps, vrec) in enumerate(vdb_rows):
        hpart = ""
        if hnsw_rows and i < len(hnsw_rows):
            _, hqps, hrec = hnsw_rows[i]
            hpart = f"  {hqps:>10,.0f} {hrec:>6.4f}"
        fpart = ""
        if faiss_rows and i < len(faiss_rows):
            _, fqps, frec = faiss_rows[i]
            fpart = f"  {fqps:>10,.0f} {frec:>6.4f}"
        print(f"{ef:>5}  {vqps:>10,.0f} {vrec:>6.4f}{hpart}{fpart}")

    print()
    parts = [f"VectorDB: {vdb_build:.1f}s"]
    if hnsw_rows:
        parts.append(f"hnswlib: {hnsw_build:.1f}s")
    if faiss_rows:
        parts.append(f"faiss: {faiss_build:.1f}s")
    print("Build — " + "  ".join(parts))

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(8, 5))
        v_recs = [r for _, _, r in vdb_rows]
        v_qps  = [q for _, q, _ in vdb_rows]
        ax.plot(v_recs, v_qps, "o-", color="tab:blue", label="VectorDB")
        for ef, qps, rec in vdb_rows:
            ax.annotate(f"ef={ef}", (rec, qps),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)

        if hnsw_rows:
            h_recs = [r for _, _, r in hnsw_rows]
            h_qps  = [q for _, q, _ in hnsw_rows]
            ax.plot(h_recs, h_qps, "s-", color="tab:orange", label="hnswlib")
            for ef, qps, rec in hnsw_rows:
                ax.annotate(f"ef={ef}", (rec, qps),
                            textcoords="offset points", xytext=(4, -12), fontsize=7)

        if faiss_rows:
            f_recs = [r for _, _, r in faiss_rows]
            f_qps  = [q for _, q, _ in faiss_rows]
            ax.plot(f_recs, f_qps, "^-", color="tab:green", label="faiss")
            for ef, qps, rec in faiss_rows:
                ax.annotate(f"ef={ef}", (rec, qps),
                            textcoords="offset points", xytext=(-30, 4), fontsize=7)

        ax.set_xlabel("Recall@10")
        ax.set_ylabel("QPS")
        ax.set_title(f"GloVe-1.2M: QPS vs Recall@10  (M=16, ef_construction=200, cosine)\n"
                     f"{N:,} vectors, dim={dim}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = os.path.join(args.out, "qps_vs_recall10.png")
        fig.savefig(plot_path, dpi=150)
        print(f"Plot saved to {plot_path}")
        plt.close()


if __name__ == "__main__":
    main()
