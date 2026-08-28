"""Reproducible GloVe ANN comparison: VectorDB vs hnswlib vs Faiss."""
import argparse
import json
import os
import shutil
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import vectordb

from bench_ann_compare import (
    HAS_FAISS, HAS_HNSWLIB, HAS_MATPLOTLIB, atomic_json_dump,
    environment_manifest, faiss, hnswlib, plt, summary, timed_search,
)


def normalize(rows):
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.ascontiguousarray(rows / np.where(norms == 0, 1, norms),
                                dtype=np.float32)


def exact_cosine_ground_truth(train, test, k=10):
    train_n = normalize(train)
    test_n = normalize(test)
    if HAS_FAISS:
        exact = faiss.IndexFlatIP(train.shape[1])
        exact.add(train_n)
        return exact.search(test_n, k)[1]

    neighbors = np.empty((len(test), k), dtype=np.int64)
    for start in range(0, len(test), 128):
        scores = test_n[start:start + 128] @ train_n.T
        candidates = np.argpartition(scores, -k, axis=1)[:, -k:]
        candidate_scores = np.take_along_axis(scores, candidates, axis=1)
        order = np.argsort(-candidate_scores, axis=1)
        neighbors[start:start + len(scores)] = np.take_along_axis(
            candidates, order, axis=1)
    return neighbors


def run_vectordb(train, test, gt, args):
    n, dim = train.shape
    build_samples = []
    db = None
    print("  [VectorDB] building index")
    for repeat in range(args.build_repeats):
        db_dir = os.path.join(args.out, "db", f"vdb-build-{repeat}")
        shutil.rmtree(db_dir, ignore_errors=True)
        db = vectordb.open(db_dir)
        db.create_collection("glove", dimension=dim, metric="cosine")
        started = time.perf_counter()
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            db.insert("glove", ids=list(range(start, end)),
                      vectors=train[start:end], num_threads=args.threads)
        elapsed = time.perf_counter() - started
        build_samples.append(elapsed)
        print(f"    repeat {repeat + 1}: {elapsed:.3f}s ({n / elapsed:,.0f} vec/s)")
        if repeat + 1 < args.build_repeats:
            del db
            shutil.rmtree(db_dir, ignore_errors=True)

    rows = timed_search(
        lambda queries, ef: db.search_batch(
            "glove", queries=queries, top_k=10, ef_search=ef,
            num_threads=args.threads),
        lambda batch: [[int(result["id"]) for result in row] for row in batch],
        test, gt, args.ef_values, args.repeats, args.warmup_queries)
    return {"build_seconds": summary(build_samples), "search": rows}


def run_hnswlib(train, test, gt, args):
    if not HAS_HNSWLIB:
        return None
    n, dim = train.shape
    build_samples = []
    index = None
    print("  [hnswlib] building index")
    for repeat in range(args.build_repeats):
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=n, ef_construction=args.ef_construction,
                         M=args.M)
        index.set_num_threads(args.threads)
        started = time.perf_counter()
        index.add_items(train, np.arange(n), num_threads=args.threads)
        elapsed = time.perf_counter() - started
        build_samples.append(elapsed)
        print(f"    repeat {repeat + 1}: {elapsed:.3f}s ({n / elapsed:,.0f} vec/s)")

    rows = timed_search(
        lambda queries, ef: (
            index.set_ef(ef),
            index.knn_query(queries, k=10, num_threads=args.threads)[0]
        )[1],
        lambda labels: labels.tolist(), test, gt, args.ef_values,
        args.repeats, args.warmup_queries)
    return {"build_seconds": summary(build_samples), "search": rows}


def run_faiss(train, test, gt, args):
    if not HAS_FAISS:
        return None
    n, dim = train.shape
    train_n = normalize(train)
    test_n = normalize(test)
    faiss.omp_set_num_threads(args.threads)
    build_samples = []
    index = None
    print("  [Faiss] building index")
    for repeat in range(args.build_repeats):
        index = faiss.IndexHNSWFlat(dim, args.M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = args.ef_construction
        started = time.perf_counter()
        index.add(train_n)
        elapsed = time.perf_counter() - started
        build_samples.append(elapsed)
        print(f"    repeat {repeat + 1}: {elapsed:.3f}s ({n / elapsed:,.0f} vec/s)")

    rows = timed_search(
        lambda queries, ef: (
            setattr(index.hnsw, "efSearch", ef),
            index.search(queries, 10)[1]
        )[1],
        lambda labels: labels.tolist(), test_n, gt, args.ef_values,
        args.repeats, args.warmup_queries)
    return {"build_seconds": summary(build_samples), "search": rows}


def plot_results(path, systems, args, n, dim):
    if not HAS_MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        "VectorDB": ("o-", "tab:blue"),
        "hnswlib": ("s-", "tab:orange"),
        "Faiss": ("^-", "tab:green"),
    }
    for name, result in systems.items():
        if result is None:
            continue
        style, color = styles[name]
        ax.plot([row["recall_at_10"] for row in result["search"]],
                [row["qps"]["median"] for row in result["search"]],
                style, color=color, label=name)
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Median QPS")
    ax.set_title(
        f"GloVe ANN comparison (M={args.M}, efConstruction={args.ef_construction}, "
        f"threads={args.threads})\n{n:,} vectors, dim={dim}, {args.repeats} search runs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/glove-200-angular.hdf5")
    parser.add_argument("--n-base", type=int, default=1_183_514)
    parser.add_argument("--n-queries", type=int, default=10_000)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-values", type=int, nargs="+",
                        default=[50, 100, 200, 400, 800])
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--build-repeats", type=int, default=1)
    parser.add_argument("--warmup-queries", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--out", default="bench_results/glove_compare")
    args = parser.parse_args()

    if args.M != 16 or args.ef_construction != 200:
        parser.error("VectorDB currently fixes M=16 and efConstruction=200; "
                     "non-default values would make the comparison unfair")
    if min(args.threads, args.repeats, args.build_repeats) <= 0:
        parser.error("threads, repeats, and build-repeats must be positive")

    os.makedirs(args.out, exist_ok=True)
    data_path = os.path.join(os.path.dirname(__file__), "..", args.data)
    print(f"Loading GloVe from {data_path}")
    with h5py.File(data_path, "r") as handle:
        full_base_count = handle["train"].shape[0]
        train = handle["train"][:args.n_base].astype(np.float32)
        test = handle["test"][:args.n_queries].astype(np.float32)
        gt = handle["neighbors"][:args.n_queries]
    if len(train) < full_base_count:
        print("  subset selected: recomputing exact cosine ground truth")
        gt = exact_cosine_ground_truth(train, test)
    print(f"  base={len(train):,} queries={len(test):,} dim={train.shape[1]} "
          f"threads={args.threads} search_repeats={args.repeats}\n")

    systems = {
        "VectorDB": run_vectordb(train, test, gt, args),
        "hnswlib": run_hnswlib(train, test, gt, args),
        "Faiss": run_faiss(train, test, gt, args),
    }
    payload = {
        "schema_version": 1,
        "benchmark": "glove-ann-comparison",
        "environment": environment_manifest(),
        "config": {
            "dataset": os.path.abspath(data_path),
            "n_base": len(train),
            "n_queries": len(test),
            "dimension": int(train.shape[1]),
            "metric": "cosine",
            "M": args.M,
            "ef_construction": args.ef_construction,
            "ef_values": args.ef_values,
            "threads": args.threads,
            "search_repeats": args.repeats,
            "build_repeats": args.build_repeats,
            "warmup_queries": args.warmup_queries,
            "batch_size": args.batch_size,
            "system_order": list(systems),
        },
        "systems": systems,
    }
    results_path = os.path.join(args.out, "results.json")
    atomic_json_dump(results_path, payload)
    plot_results(os.path.join(args.out, "qps_vs_recall10.png"),
                 systems, args, len(train), train.shape[1])
    print(f"\nMachine-readable results written to {results_path}")


if __name__ == "__main__":
    main()
