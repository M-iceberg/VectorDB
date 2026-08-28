"""Reproducible SIFT ANN comparison: VectorDB vs hnswlib vs Faiss.

The benchmark pins an explicit thread count, warms every search path, records
all timing samples, and atomically writes a machine-readable results manifest.
QPS is reported as the median of repeated full-query runs.
"""
import argparse
import datetime as dt
import importlib.metadata
import json
import os
import platform
import shutil
import statistics
import subprocess
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

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def recall_at_10(returned, gt):
    found = 0
    total = 0
    for i, result in enumerate(returned):
        expected = set(gt[i, :10].tolist())
        found += len(expected & set(result[:10]))
        total += len(expected)
    return found / total if total else 0.0


def exact_l2_ground_truth(train, test, k=10):
    """Compute valid ground truth when benchmarking a dataset subset."""
    if HAS_FAISS:
        exact = faiss.IndexFlatL2(train.shape[1])
        exact.add(train)
        return exact.search(test, k)[1]

    neighbors = np.empty((len(test), k), dtype=np.int64)
    train_norm = np.sum(train * train, axis=1)
    for start in range(0, len(test), 128):
        queries = test[start:start + 128]
        distances = (np.sum(queries * queries, axis=1, keepdims=True) +
                     train_norm[None, :] - 2.0 * queries @ train.T)
        candidates = np.argpartition(distances, k - 1, axis=1)[:, :k]
        candidate_distances = np.take_along_axis(distances, candidates, axis=1)
        order = np.argsort(candidate_distances, axis=1)
        neighbors[start:start + len(queries)] = np.take_along_axis(
            candidates, order, axis=1)
    return neighbors


def summary(samples):
    values = [float(v) for v in samples]
    return {
        "samples": values,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def package_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_dirty():
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def compiler_version():
    try:
        return subprocess.check_output(
            [os.environ.get("CXX", "c++"), "--version"], text=True,
            stderr=subprocess.STDOUT).splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def cpu_details():
    """Return enough CPU identity/SIMD data to audit cross-machine results."""
    details = {"model": "unknown", "flags": []}
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                fields = {}
                for line in handle:
                    if not line.strip():
                        break
                    if ":" in line:
                        key, value = line.split(":", 1)
                        fields[key.strip()] = value.strip()
            details["model"] = fields.get("model name", fields.get("Processor", "unknown"))
            details["flags"] = fields.get("flags", fields.get("Features", "")).split()
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            details["model"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
                stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            details["model"] = platform.processor() or "unknown"
    return details


def environment_manifest():
    manifest = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "command": sys.argv,
        "compiler": compiler_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "numpy": np.__version__,
        "vectordb": package_version("vectordb"),
        "hnswlib": package_version("hnswlib") if HAS_HNSWLIB else None,
        "faiss": package_version("faiss-cpu") if HAS_FAISS else None,
    }
    manifest["cpu"] = cpu_details()
    return manifest


def atomic_json_dump(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def timed_search(search_fn, convert_fn, test, gt, ef_values, repeats, warmup):
    rows = []
    warmup_queries = test[:min(warmup, len(test))]
    for ef in ef_values:
        search_fn(warmup_queries, ef)
        elapsed_samples = []
        returned = None
        for _ in range(repeats):
            started = time.perf_counter()
            raw = search_fn(test, ef)
            elapsed_samples.append(time.perf_counter() - started)
            returned = convert_fn(raw)
        qps_samples = [len(test) / elapsed for elapsed in elapsed_samples]
        recall = recall_at_10(returned, gt)
        row = {
            "ef_search": ef,
            "qps": summary(qps_samples),
            "elapsed_seconds": summary(elapsed_samples),
            "recall_at_10": recall,
        }
        rows.append(row)
        print(f"    ef={ef:>4}  median QPS={row['qps']['median']:>9,.0f}  "
              f"range=[{row['qps']['min']:,.0f}, {row['qps']['max']:,.0f}]  "
              f"R@10={recall:.4f}")
    return rows


def run_vectordb(train, test, gt, args):
    n, dim = train.shape
    build_samples = []
    db = None
    os.environ["VECTORDB_STORAGE_MODE"] = args.vectordb_storage_mode
    print(f"  [VectorDB/{args.vectordb_storage_mode}] building index")
    for repeat in range(args.build_repeats):
        db_dir = os.path.join(args.out, "db", f"vdb-build-{repeat}")
        shutil.rmtree(db_dir, ignore_errors=True)
        db = vectordb.open(db_dir)
        db.create_collection("sift", dimension=dim, metric="l2")
        started = time.perf_counter()
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            db.insert("sift", ids=list(range(start, end)),
                      vectors=train[start:end], num_threads=args.threads)
        elapsed = time.perf_counter() - started
        build_samples.append(elapsed)
        print(f"    repeat {repeat + 1}: {elapsed:.3f}s ({n / elapsed:,.0f} vec/s)")
        if repeat + 1 < args.build_repeats:
            del db
            shutil.rmtree(db_dir, ignore_errors=True)

    rows = timed_search(
        lambda queries, ef: db.search_batch(
            "sift", queries=queries, top_k=10, ef_search=ef,
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
        index = hnswlib.Index(space="l2", dim=dim)
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
    faiss.omp_set_num_threads(args.threads)
    build_samples = []
    index = None
    print("  [Faiss] building index")
    for repeat in range(args.build_repeats):
        index = faiss.IndexHNSWFlat(dim, args.M)
        index.hnsw.efConstruction = args.ef_construction
        started = time.perf_counter()
        index.add(train)
        elapsed = time.perf_counter() - started
        build_samples.append(elapsed)
        print(f"    repeat {repeat + 1}: {elapsed:.3f}s ({n / elapsed:,.0f} vec/s)")

    rows = timed_search(
        lambda queries, ef: (
            setattr(index.hnsw, "efSearch", ef),
            index.search(queries, 10)[1]
        )[1],
        lambda labels: labels.tolist(), test, gt, args.ef_values,
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
        rows = result["search"]
        style, color = styles[name]
        ax.plot([row["recall_at_10"] for row in rows],
                [row["qps"]["median"] for row in rows],
                style, color=color, label=name)
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Median QPS")
    ax.set_title(
        f"SIFT ANN comparison (M={args.M}, efConstruction={args.ef_construction}, "
        f"threads={args.threads})\n{n:,} vectors, dim={dim}, {args.repeats} search runs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sift-128-euclidean.hdf5")
    parser.add_argument("--n-base", type=int, default=1_000_000)
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
    parser.add_argument("--vectordb-storage-mode",
                        choices=("performance", "compact"),
                        default="performance")
    parser.add_argument("--out", default="bench_results/ann_compare")
    args = parser.parse_args()

    if args.M != 16 or args.ef_construction != 200:
        parser.error("VectorDB currently fixes M=16 and efConstruction=200; "
                     "non-default values would make the comparison unfair")
    if min(args.threads, args.repeats, args.build_repeats) <= 0:
        parser.error("threads, repeats, and build-repeats must be positive")

    os.makedirs(args.out, exist_ok=True)
    data_path = os.path.join(os.path.dirname(__file__), "..", args.data)
    print(f"Loading SIFT from {data_path}")
    with h5py.File(data_path, "r") as handle:
        full_base_count = handle["train"].shape[0]
        train = handle["train"][:args.n_base].astype(np.float32)
        test = handle["test"][:args.n_queries].astype(np.float32)
        gt = handle["neighbors"][:args.n_queries]
    if len(train) < full_base_count:
        print("  subset selected: recomputing exact L2 ground truth")
        gt = exact_l2_ground_truth(train, test)
    print(f"  base={len(train):,} queries={len(test):,} dim={train.shape[1]} "
          f"threads={args.threads} search_repeats={args.repeats}\n")

    systems = {
        "VectorDB": run_vectordb(train, test, gt, args),
        "hnswlib": run_hnswlib(train, test, gt, args),
        "Faiss": run_faiss(train, test, gt, args),
    }
    payload = {
        "schema_version": 1,
        "benchmark": "sift-ann-comparison",
        "environment": environment_manifest(),
        "config": {
            "dataset": os.path.abspath(data_path),
            "n_base": len(train),
            "n_queries": len(test),
            "dimension": int(train.shape[1]),
            "metric": "l2",
            "M": args.M,
            "ef_construction": args.ef_construction,
            "ef_values": args.ef_values,
            "threads": args.threads,
            "search_repeats": args.repeats,
            "build_repeats": args.build_repeats,
            "warmup_queries": args.warmup_queries,
            "batch_size": args.batch_size,
            "vectordb_storage_mode": args.vectordb_storage_mode,
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
