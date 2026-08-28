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

from bench_ann_compare import atomic_json_dump, environment_manifest, summary

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
threads   = int(sys.argv[5])

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
    db.insert('sift', ids=list(range(start, end)), vectors=train[start:end],
              num_threads=threads)

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
threads   = int(sys.argv[5])

with open(data_file, 'rb') as f:
    import pickle
    train, test = pickle.load(f)

N, dim = train.shape
proc = psutil.Process()
rss_before = proc.memory_info().rss

index = hnswlib.Index(space='l2', dim=dim)
index.init_index(max_elements=N, ef_construction=200, M=16)
index.set_num_threads(threads)
index.add_items(train, list(range(N)), num_threads=threads)

rss_after = proc.memory_info().rss
index_mb = (rss_after - rss_before) / (1024*1024)

index.set_ef(ef)
def q1(q): index.knn_query(q.reshape(1,-1), k=10, num_threads=1)
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
threads   = int(sys.argv[5])

with open(data_file, 'rb') as f:
    import pickle
    train, test = pickle.load(f)

N, dim = train.shape
proc = psutil.Process()
rss_before = proc.memory_info().rss

index = faiss.IndexHNSWFlat(dim, 16)
index.hnsw.efConstruction = 200
faiss.omp_set_num_threads(threads)
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


def run_worker(script_src, data_pkl, n_base, ef, out_dir, name, threads,
               storage_mode=None):
    import json, pickle, tempfile

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(script_src)
        script_path = f.name

    try:
        env = os.environ.copy()
        python_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
        env["PYTHONPATH"] = python_dir + os.pathsep + env.get("PYTHONPATH", "")
        if storage_mode:
            env["VECTORDB_STORAGE_MODE"] = storage_mode
        result = subprocess.run(
            [sys.executable, script_path, data_pkl, str(n_base), str(ef),
             out_dir, str(threads)],
            capture_output=True, text=True, timeout=1200, env=env
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
    parser.add_argument("--threads",   type=int, default=os.cpu_count() or 1)
    parser.add_argument("--repeats",   type=int, default=3)
    parser.add_argument("--vectordb-modes", nargs="+",
                        choices=("performance", "compact"),
                        default=("performance",),
                        help="VectorDB storage modes to measure")
    parser.add_argument("--out",       default="bench_results/memory_latency")
    args = parser.parse_args()
    if args.threads <= 0 or args.repeats <= 0:
        parser.error("threads and repeats must be positive")

    os.makedirs(args.out, exist_ok=True)
    data_path = os.path.join(os.path.dirname(__file__), "..", args.data)

    print(f"Loading SIFT from {data_path} ...")
    with h5py.File(data_path, "r") as f:
        train = f["train"][:args.n_base].astype(np.float32)
        test  = f["test"][:args.n_queries].astype(np.float32)
    N, dim = train.shape
    print(f"  {N:,} base  {len(test):,} queries  dim={dim}  ef={args.ef}  "
          f"threads={args.threads}  repeats={args.repeats}\n")

    # Serialize data once so workers can load it quickly
    import pickle
    data_pkl = os.path.join(args.out, "data.pkl")
    print("Saving data for worker subprocesses ...")
    with open(data_pkl, "wb") as f:
        pickle.dump((train, test), f)
    print()

    raw_mb = N * dim * 4 / (1024 * 1024)

    worker_specs = [
        (f"VectorDB-{mode}", _WORKER_VECTORDB, mode)
        for mode in args.vectordb_modes
    ] + [
        ("hnswlib", _WORKER_HNSWLIB, None),
        ("Faiss", _WORKER_FAISS, None),
    ]
    results = {}
    try:
        for name, worker, storage_mode in worker_specs:
            print(f"=== {name} ===  (clean subprocess per repeat)")
            runs = []
            for repeat in range(args.repeats):
                run = run_worker(worker, data_pkl, args.n_base, args.ef,
                                 args.out, name, args.threads, storage_mode)
                if run is None:
                    raise RuntimeError(f"{name} repeat {repeat + 1} failed")
                runs.append(run)
                print(f"  repeat {repeat + 1}: RSS={run['index_mb']:.0f} MB  "
                      f"P50={run['p50']:.3f}ms P95={run['p95']:.3f}ms "
                      f"P99={run['p99']:.3f}ms")
            results[name] = {
                "index_mb": summary([run["index_mb"] for run in runs]),
                "p50_ms": summary([run["p50"] for run in runs]),
                "p95_ms": summary([run["p95"] for run in runs]),
                "p99_ms": summary([run["p99"] for run in runs]),
                "raw_runs": runs,
            }
            print()
    finally:
        if os.path.exists(data_pkl):
            os.unlink(data_pkl)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"Raw vector data (no index): {raw_mb:.0f} MB  ({dim * 4} B/vec)\n")
    print(f"{'System':<22}  {'Index MB':>10}  {'B/vec':>8}  {'vs raw':>8}  "
          f"  {'P50 ms':>7}  {'P95 ms':>7}  {'P99 ms':>7}")
    print("-" * 82)
    for name, r in results.items():
        mb = r["index_mb"]["median"]
        bpv = mb * 1024 * 1024 / N
        ratio = mb / raw_mb
        print(f"{name:<22}  {mb:>10.0f}  {bpv:>8.0f}  {ratio:>7.1f}×  "
              f"  {r['p50_ms']['median']:>7.2f}  "
              f"{r['p95_ms']['median']:>7.2f}  {r['p99_ms']['median']:>7.2f}")

    payload = {
        "schema_version": 1,
        "benchmark": "sift-memory-latency-comparison",
        "environment": environment_manifest(),
        "config": {
            "dataset": os.path.abspath(data_path),
            "n_base": N,
            "n_queries": len(test),
            "dimension": dim,
            "metric": "l2",
            "M": 16,
            "ef_construction": 200,
            "ef_search": args.ef,
            "threads_for_build": args.threads,
            "threads_per_query": 1,
            "warmup_queries": min(200, len(test)),
            "repeats": args.repeats,
            "vectordb_storage_modes": list(args.vectordb_modes),
            "raw_vector_mb": raw_mb,
        },
        "systems": results,
    }
    results_path = os.path.join(args.out, "results.json")
    atomic_json_dump(results_path, payload)
    print(f"\nMachine-readable results written to {results_path}")


if __name__ == "__main__":
    main()
