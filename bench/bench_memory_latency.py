"""
Memory usage and per-query latency comparison: VectorDB vs hnswlib vs faiss.

Builds a SIFT-1M index for each implementation, measures:
  - Index peak RSS (each implementation in its own subprocess for clean measurement)
  - Per-query latency: P50, P95, P99 at ef=200 (single-threaded, one query at a time)

Usage:
    python bench/bench_memory_latency.py [--data data/sift-128-euclidean.hdf5]
    python bench/bench_memory_latency.py --n-base 100000   # quick run on subset

Requirements:
    pip install hnswlib faiss-cpu h5py psutil
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


# ---------------------------------------------------------------------------
# Worker scripts run in subprocesses for clean RSS measurement
# ---------------------------------------------------------------------------

_WORKER_VECTORDB = """
import sys, os, gc, time, json, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__) if '__file__' in dir() else '.', '..', 'python'))
import numpy as np, psutil, vectordb

data_file = sys.argv[1]
n_base    = int(sys.argv[2])
ef        = int(sys.argv[3])
out_dir   = sys.argv[4]

with open(data_file, 'rb') as f:
    import pickle
    train, test = pickle.load(f)

N, dim = train.shape
db_dir = os.path.join(out_dir, 'vdb')
shutil.rmtree(db_dir, ignore_errors=True)

proc = psutil.Process()
rss_before = proc.memory_info().rss

db = vectordb.open(db_dir)
db.create_collection('sift', dimension=dim, metric='l2')
BATCH = 10_000
for start in range(0, N, BATCH):
    end = min(start + BATCH, N)
    db.insert('sift', ids=list(range(start, end)), vectors=train[start:end])

rss_after = proc.memory_info().rss
index_mb = (rss_after - rss_before) / (1024*1024)

# latency
def q1(q):
    db.search('sift', query=q, top_k=10, ef_search=ef)
for q in test[:200]: q1(q)  # warmup
times = []
for q in test:
    t0 = time.perf_counter()
    q1(q)
    times.append(time.perf_counter() - t0)

times.sort()
n = len(times)
result = dict(
    index_mb=index_mb,
    p50=times[int(n*0.50)]*1000,
    p95=times[int(n*0.95)]*1000,
    p99=times[int(n*0.99)]*1000,
)
print(json.dumps(result))
"""

_WORKER_HNSWLIB = """
import sys, os, gc, time, json
import numpy as np, psutil, hnswlib

data_file = sys.argv[1]
n_base    = int(sys.argv[2])
ef        = int(sys.argv[3])

with open(data_file, 'rb') as f:
    import pickle
    train, test = pickle.load(f)

N, dim = train.shape
proc = psutil.Process()
rss_before = proc.memory_info().rss

index = hnswlib.Index(space='l2', dim=dim)
index.init_index(max_elements=N, ef_construction=200, M=16)
index.add_items(train, list(range(N)))

rss_after = proc.memory_info().rss
index_mb = (rss_after - rss_before) / (1024*1024)

index.set_ef(ef)
def q1(q): index.knn_query(q.reshape(1,-1), k=10)
for q in test[:200]: q1(q)
times = []
for q in test:
    t0 = time.perf_counter()
    q1(q)
    times.append(time.perf_counter() - t0)

times.sort()
n = len(times)
result = dict(
    index_mb=index_mb,
    p50=times[int(n*0.50)]*1000,
    p95=times[int(n*0.95)]*1000,
    p99=times[int(n*0.99)]*1000,
)
print(json.dumps(result))
"""

_WORKER_FAISS = """
import sys, os, gc, time, json
import numpy as np, psutil, faiss

data_file = sys.argv[1]
n_base    = int(sys.argv[2])
ef        = int(sys.argv[3])

with open(data_file, 'rb') as f:
    import pickle
    train, test = pickle.load(f)

N, dim = train.shape
proc = psutil.Process()
rss_before = proc.memory_info().rss

index = faiss.IndexHNSWFlat(dim, 16)
index.hnsw.efConstruction = 200
index.add(train)

rss_after = proc.memory_info().rss
index_mb = (rss_after - rss_before) / (1024*1024)

index.hnsw.efSearch = ef
def q1(q): index.search(q.reshape(1,-1), 10)
for q in test[:200]: q1(q)
times = []
for q in test:
    t0 = time.perf_counter()
    q1(q)
    times.append(time.perf_counter() - t0)

times.sort()
n = len(times)
result = dict(
    index_mb=index_mb,
    p50=times[int(n*0.50)]*1000,
    p95=times[int(n*0.95)]*1000,
    p99=times[int(n*0.99)]*1000,
)
print(json.dumps(result))
"""


def run_worker(script_src, data_pkl, n_base, ef, out_dir, name):
    import json, pickle, tempfile

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(script_src)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path, data_pkl, str(n_base), str(ef), out_dir],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            print(f"  [{name}] FAILED:\n{result.stderr[-800:]}")
            return None
        last_line = [l for l in result.stdout.strip().splitlines() if l.startswith("{")]
        if not last_line:
            print(f"  [{name}] no JSON output. stdout:\n{result.stdout[-400:]}")
            return None
        return json.loads(last_line[-1])
    finally:
        os.unlink(script_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      default="data/sift-128-euclidean.hdf5")
    parser.add_argument("--n-base",    type=int, default=1_000_000)
    parser.add_argument("--n-queries", type=int, default=2_000)
    parser.add_argument("--ef",        type=int, default=200)
    parser.add_argument("--out",       default="bench_results/memory_latency")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data_path = os.path.join(os.path.dirname(__file__), "..", args.data)

    print(f"Loading SIFT from {data_path} ...")
    with h5py.File(data_path, "r") as f:
        train = f["train"][:args.n_base].astype(np.float32)
        test  = f["test"][:args.n_queries].astype(np.float32)
    N, dim = train.shape
    print(f"  {N:,} base  {len(test):,} queries  dim={dim}  ef={args.ef}\n")

    # Serialize data once so workers can load it quickly
    import pickle
    data_pkl = os.path.join(args.out, "data.pkl")
    print("Saving data for worker subprocesses ...")
    with open(data_pkl, "wb") as f:
        pickle.dump((train, test), f)
    print()

    raw_mb = N * dim * 4 / (1024 * 1024)

    results = {}

    print("=== VectorDB ===  (building in subprocess ...)")
    r = run_worker(_WORKER_VECTORDB, data_pkl, args.n_base, args.ef, args.out, "VectorDB")
    if r:
        results["VectorDB"] = r
        print(f"  Index RSS: {r['index_mb']:.0f} MB  ({r['index_mb']/raw_mb:.1f}× raw)")
        print(f"  Latency   P50={r['p50']:.2f}ms  P95={r['p95']:.2f}ms  P99={r['p99']:.2f}ms\n")

    print("=== hnswlib ===  (building in subprocess ...)")
    r = run_worker(_WORKER_HNSWLIB, data_pkl, args.n_base, args.ef, args.out, "hnswlib")
    if r:
        results["hnswlib"] = r
        print(f"  Index RSS: {r['index_mb']:.0f} MB  ({r['index_mb']/raw_mb:.1f}× raw)")
        print(f"  Latency   P50={r['p50']:.2f}ms  P95={r['p95']:.2f}ms  P99={r['p99']:.2f}ms\n")

    print("=== faiss ===  (building in subprocess ...)")
    r = run_worker(_WORKER_FAISS, data_pkl, args.n_base, args.ef, args.out, "faiss")
    if r:
        results["faiss"] = r
        print(f"  Index RSS: {r['index_mb']:.0f} MB  ({r['index_mb']/raw_mb:.1f}× raw)")
        print(f"  Latency   P50={r['p50']:.2f}ms  P95={r['p95']:.2f}ms  P99={r['p99']:.2f}ms\n")

    os.unlink(data_pkl)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"Raw vector data (no index): {raw_mb:.0f} MB  ({dim * 4} B/vec)\n")
    print(f"{'System':<12}  {'Index MB':>10}  {'B/vec':>8}  {'vs raw':>8}  "
          f"  {'P50 ms':>7}  {'P95 ms':>7}  {'P99 ms':>7}")
    print("-" * 72)
    for name, r in results.items():
        mb = r["index_mb"]
        bpv = mb * 1024 * 1024 / N
        ratio = mb / raw_mb
        print(f"{name:<12}  {mb:>10.0f}  {bpv:>8.0f}  {ratio:>7.1f}×  "
              f"  {r['p50']:>7.2f}  {r['p95']:>7.2f}  {r['p99']:>7.2f}")


if __name__ == "__main__":
    main()
