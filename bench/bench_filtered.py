"""
Filtered search benchmark: QPS and recall under different selectivity levels.

Inserts N vectors each tagged with a category from {0..K-1}, so
selectivity = 1/K. Measures filtered search QPS and recall vs
brute-force ground truth for each selectivity.

Usage:
    python bench/bench_filtered.py [--n 10000] [--dim 128] [--queries 500]
    python bench/bench_filtered.py --ci
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import vectordb

# (selectivity_pct, n_categories)  selectivity = 1/K
SELECTIVITIES = [
    (100.0,   1),   # no filter effect
    ( 50.0,   2),
    ( 10.0,  10),
    (  1.0, 100),
    (  0.1, 1000),
]


def brute_force_filtered(vecs: np.ndarray, query: np.ndarray,
                         labels: np.ndarray, target_label: int,
                         top_k: int) -> list[int]:
    mask = labels == target_label
    idx  = np.where(mask)[0]
    if len(idx) == 0:
        return []
    dists = np.sum((vecs[idx] - query) ** 2, axis=1)
    order = np.argsort(dists)[:top_k]
    return idx[order].tolist()


def run_selectivity(sel_pct: float, n_categories: int,
                    vecs: np.ndarray, queries: np.ndarray,
                    db, dim: int, top_k: int) -> dict:
    n = len(vecs)
    labels = np.arange(n) % n_categories  # uniform distribution

    # If the DB already has a collection, drop it and rebuild.
    try:
        db.drop_collection("col")
    except Exception:
        pass
    db.create_collection("col", dimension=dim, metric="l2")

    # Insert with category metadata.
    BATCH = 500
    for start in range(0, n, BATCH):
        batch = min(BATCH, n - start)
        ids   = list(range(start, start + batch))
        v     = vecs[start:start + batch]
        meta  = [{"category": str(labels[start + i])} for i in range(batch)]
        db.insert("col", ids=ids, vectors=v, metadata=meta)

    # Measure QPS over all queries.
    q_vecs  = queries
    n_q     = len(q_vecs)
    # Each query targets category 0 (contains n / n_categories vectors).
    filter_ = {"category": "0"}

    t0 = time.perf_counter()
    all_results = []
    for q in q_vecs:
        r = db.search("col", query=q, top_k=top_k, ef_search=100, filters=filter_)
        all_results.append({x["id"] for x in r})
    elapsed = time.perf_counter() - t0
    qps = n_q / elapsed

    # Compute recall@top_k vs brute-force.
    hits = 0
    for i, q in enumerate(q_vecs):
        gt = brute_force_filtered(vecs, q, labels, 0, top_k)
        gt_ids = {str(g) for g in gt}
        if gt_ids & all_results[i]:
            hits += 1
    recall = hits / n_q if n_q > 0 else 0.0

    return dict(selectivity_pct=sel_pct, n_categories=n_categories,
                n_filtered=n // n_categories, qps=qps, recall=recall)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int, default=10_000)
    parser.add_argument("--dim",     type=int, default=128)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--top-k",   type=int, default=10)
    parser.add_argument("--out",     default="bench_results/filtered")
    parser.add_argument("--ci",      action="store_true",
                        help="Smaller N and query count for CI")
    args = parser.parse_args()

    if args.ci:
        n = 3_000
        dim = 32
        n_queries = 50
        sels = SELECTIVITIES[:-1]  # skip 0.1% (only 3 vectors in that bucket)
    else:
        n = args.n
        dim = args.dim
        n_queries = args.queries
        sels = SELECTIVITIES

    os.makedirs(args.out, exist_ok=True)
    db_dir = os.path.join(args.out, "db")
    shutil.rmtree(db_dir, ignore_errors=True)

    rng     = np.random.default_rng(42)
    vecs    = rng.random((n, dim), dtype=np.float32)
    queries = rng.random((n_queries, dim), dtype=np.float32)

    db = vectordb.open(db_dir)

    print(f"Filtered search benchmark  n={n:,}  dim={dim}  "
          f"queries={n_queries}  top_k={args.top_k}")
    print()
    print(f"{'Selectivity':>12} {'#Candidates':>12} {'QPS':>8} {'Recall@10':>10}")
    print("-" * 48)

    rows = []
    for sel_pct, n_cat in sels:
        r = run_selectivity(sel_pct, n_cat, vecs, queries, db, dim, args.top_k)
        rows.append(r)
        print(f"{r['selectivity_pct']:>11.1f}% {r['n_filtered']:>12,} "
              f"{r['qps']:>8.0f} {r['recall']:>10.3f}")

    print()
    print("Selectivity = fraction of vectors passing the filter.")
    print("At low selectivity (1%) HNSW visits many nodes that are then rejected,")
    print("so recall drops. Filtered ANN is fundamentally harder than unfiltered.")

    import json
    summary = dict(n=n, dim=dim, queries=n_queries, top_k=args.top_k, results=rows)
    path = os.path.join(args.out, "results.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
