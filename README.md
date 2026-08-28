# VectorDB

A persistent single-node vector search engine built from scratch in C++17,
with Python bindings and reproducible ANN benchmarks.

**Core stack:** HNSW · NEON/AVX2 · mmap · CRC-protected WAL · crash-atomic
checkpoints · metadata filtering · pybind11

**SIFT-1M:** median **40.2K QPS at 99.64% Recall@10** and **40.8K
vectors/s** build throughput (`M=16`, `efConstruction=200`, `efSearch=200`,
18 threads, three runs). At this operating point VectorDB measured **1.44×
hnswlib QPS** and **2.15× Faiss QPS**. Three clean-process latency runs at
the same `efSearch=200` measured **0.296 ms median P99** in performance mode,
versus 0.567 ms for hnswlib and 0.479 ms for Faiss.

---

## Architecture

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                        Python SDK                               │
  │         open()  insert()  search()  remove()  checkpoint()      │
  └──────────────────────────┬──────────────────────────────────────┘
                             │ pybind11 (copy-free for C-contiguous float32)
  ┌──────────────────────────▼──────────────────────────────────────┐
  │                          Engine                                 │
  │              shared_mutex: readers concurrent, writes exclusive  │
  │                                                                  │
  │   insert_batch()  ───────────────────────────────────────────►  │
  │     │  1 ▸ reserve stable NodeIds                                │
  │     │  2 ▸ WAL append + one group fsync       [commit point]     │
  │     │  3 ▸ parallel HNSW construction         [striped locks]    │
  │     └  4 ▸ mmap vector + metadata publication                    │
  │                                                                  │
  │   search()  ─────────────────────────────────────────────────►  │
  │     │  1 ▸ HNSW beam search (ef_search candidates)              │
  │     │  2 ▸ post-filter by metadata allowlist                     │
  │     └  3 ▸ return top-k sorted by distance                      │
  │                                                                  │
  │   Engine()  [startup — crash recovery]  ──────────────────────► │
  │     │  1 ▸ validate generation manifest + snapshot CRCs           │
  │     │  2 ▸ deserialize graph, metadata, and ID map                │
  │     └  3 ▸ replay committed WAL records                           │
  └───────┬──────────────┬──────────────┬──────────────┬────────────┘
          │              │              │              │
  ┌───────▼──────┐ ┌─────▼──────┐ ┌────▼─────┐ ┌─────▼────────────┐
  │  HnswIndex   │ │ VectorFile │ │   WAL    │ │  MetadataIndex   │
  │              │ │            │ │          │ │                  │
  │  multi-layer │ │ mmap flat  │ │ append-  │ │ inverted index   │
  │  proximity   │ │ float32    │ │ only log │ │ (string eq)      │
  │  graph       │ │ array      │ │ fdatasync│ │ sorted array     │
  │  AVX2 / NEON │ │ O(1) write │ │ per op   │ │ (numeric range)  │
  └───────┬──────┘ └─────┬──────┘ └────┬─────┘ └─────┬────────────┘
          │  checkpoint   │             │  checkpoint   │
          └───────────────┴──────┬──────┴───────────────┘
                                 │
  ┌──────────────────────────────▼──────────────────────────────────┐
  │                            Disk                                 │
  │ checkpoint.current  checkpoint-N/  vectors.vdb  wal.log          │
  └─────────────────────────────────────────────────────────────────┘
```

### Key design decisions

**HNSW** — probabilistic skip-list graph. O(log N) search, tunable recall via `ef_search`. Industry standard: hnswlib, faiss, and all major cloud vector databases use it.

**Cache-aware layer-0 layout** — each node's fixed-capacity layer-0 adjacency
list and vector are colocated in one contiguous block. Upper layers remain
variable-sized. Generation-stamped thread-local visited tables avoid clearing
or allocating a hash set per query. An optional compact mode keeps adjacency
in memory but reads the single mmap-backed vector copy directly.

**Write-ahead log** — stable IDs are reserved, then inserts/deletes are appended
and synchronously committed before graph mutation. Batch inserts use one group
commit. WAL records carry CRC32 and corrupt tails are truncated on open.

**Crash-atomic checkpoints** — immutable snapshot generations are fsynced and
published through an atomic manifest rename before WAL truncation. The recovery
suite injects SIGKILL at every publication boundary.

**Metadata as post-filter** — metadata stored in a separate in-memory index (inverted index for string equality, sorted array for numeric range). Applied after HNSW beam search, not during graph traversal. Simple to implement; breaks down at very low selectivity (<1%) — see [benchmarks](docs/benchmarks.md#filtered-search--selectivity-vs-qps-and-recall-day-28).

---

## Quickstart

```bash
pip install . --no-build-isolation
```

Requires: cmake ≥ 3.20, C++17 compiler, numpy, scikit-build-core, pybind11.

```python
import vectordb
import numpy as np

db = vectordb.open("/tmp/demo")
db.create_collection("docs", dimension=128, metric="l2")

ids  = ["doc-a", "doc-b", "doc-c"]
vecs = np.random.rand(3, 128).astype(np.float32)
db.insert("docs", ids=ids, vectors=vecs)

results = db.search("docs", query=vecs[0], top_k=2)
# [{'id': 'doc-a', 'distance': 0.0}, {'id': 'doc-c', 'distance': 14.3}]
```

### Metadata filtering

```python
db.insert("docs", ids=["p1","p2","p3"], vectors=vecs,
          metadata=[{"category":"ml","year":2024},
                    {"category":"db","year":2023},
                    {"category":"ml","year":2022}])

# string equality
results = db.search("docs", query=vecs[0], top_k=10,
                    filters={"category": "ml"})

# numeric range
results = db.search("docs", query=vecs[0], top_k=10,
                    filters={"year": {"$gte": 2023}})

# combined AND
results = db.search("docs", query=vecs[0], top_k=10,
                    filters={"category": "ml", "year": {"$gte": 2023}})
```

Supported operators: `$gte` (≥), `$lte` (≤). Multiple keys are ANDed.

### Delete and checkpoint

```python
db.remove("docs", "doc-a")     # soft-delete; never returned in search results
db.checkpoint("docs")          # snapshot to disk; truncate WAL
```

---

## Benchmarks

### SIFT-1M comparison

Apple Silicon, 18 logical CPUs, Python 3.13, hnswlib 0.8.0, Faiss 1.14.3.
All systems use `M=16`, `efConstruction=200`, the same 10K queries, explicit
18-thread batch search, 200 warm-up queries, and three measured runs.

| System | Median build | Build throughput | Median QPS at ef=200 | Recall@10 |
|---|---:|---:|---:|---:|
| **VectorDB** | **24.50 s** | **40,814 vec/s** | **40,209** | **0.9964** |
| hnswlib | 32.67 s | 30,608 vec/s | 27,882 | 0.9955 |
| Faiss | 31.65 s | 31,598 vec/s | 18,686 | 0.9960 |

VectorDB is not fastest at every operating point: Faiss leads at `efSearch=50`.
VectorDB leads both comparisons from `efSearch=100` through 800 in this run.
Use the full recall/QPS curve rather than extrapolating the ef=200 result.

The committed [raw result manifest](bench_results/ann_compare/results.json)
contains every sample, min/median/max, dependency versions, hardware, command,
thread count, and Git state. Reproduce it with:

```bash
python bench/bench_ann_compare.py \
  --threads 18 --repeats 3 --build-repeats 3
```

### Single-query latency and memory

SIFT-1M at `efSearch=200`; each system was built and measured in a fresh
process for each of three runs. Latency is one query at a time after 200
warm-up queries.

| System | Median RSS | Median P50 | Median P95 | Median P99 |
|---|---:|---:|---:|---:|
| **VectorDB performance** | 2,014 MB | **0.238 ms** | **0.280 ms** | **0.296 ms** |
| **VectorDB compact** | 1,162 MB | 0.259 ms | 0.307 ms | 0.324 ms |
| hnswlib | **820 MB** | 0.452 ms | 0.533 ms | 0.567 ms |
| Faiss | 823 MB | 0.387 ms | 0.456 ms | 0.479 ms |

Compact mode removes the HNSW-private vector copy, reducing VectorDB RSS by
42% while retaining lower median P99 than both comparison libraries. One of
three compact runs reached 0.529 ms P99; the raw data is retained rather than
reporting only the best run. The
[storage trade-off manifest](bench_results/storage_tradeoff/results.json)
records all twelve isolated-process runs. Reproduce it with:

```bash
python bench/bench_memory_latency.py \
  --threads 18 --repeats 3 \
  --vectordb-modes performance compact
```

Select the mode before opening the database:

```bash
VECTORDB_STORAGE_MODE=compact python app.py   # default: performance
```

At `efSearch=200`, compact mode also sustained 37,962 median QPS at 0.9964
Recall@10: 1.37× hnswlib and 2.06× Faiss in the same run. See the
[compact ANN manifest](bench_results/ann_compare_compact/results.json).

### Cross-dataset check

On GloVe-1.2M cosine search at `efSearch=200`, VectorDB measured 18,974 median
QPS at 0.8266 Recall@10, versus hnswlib's 14,108/0.8133 and Faiss's
11,595/0.8139. Faiss was faster to build and led search at `efSearch=50`.
The [GloVe manifest](bench_results/glove_compare/results.json) contains three
search samples per operating point and the complete environment.

Subset runs automatically recompute exact ground truth, so quick smoke results
do not accidentally use SIFT-1M neighbors for a smaller corpus. The GloVe
harness follows the same manifest and repetition format.

Historical optimization traces, memory profiling, recovery timing, and filter
selectivity experiments remain in [the benchmark notes](docs/benchmarks.md).

---

## Python SDK

| Method | Description |
|--------|-------------|
| `vectordb.open(data_dir)` | Open or create a database |
| `db.create_collection(name, dimension, metric="l2")` | Create collection. `metric`: `"l2"` \| `"cosine"` \| `"ip"` |
| `db.drop_collection(name)` | Drop collection and delete files |
| `db.list_collections()` | List all collections |
| `db.insert(col, *, ids, vectors, metadata=None, num_threads=0)` | Insert one or many vectors; `0` uses hardware concurrency |
| `db.remove(col, id)` | Soft-delete a vector by ID |
| `db.search(col, *, query, top_k=10, ef_search=64, filters=None)` | ANN search |
| `db.checkpoint(col)` | Flush snapshot to disk, truncate WAL |

Full reference: [`docs/api.md`](docs/api.md)

---

## Build from source

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Tests
./build/tests/unit/test_smoke
./build/tests/unit/test_hnsw_ut
./build/tests/unit/test_engine_recovery_int
# see .github/workflows/ci.yml for full list

# Profiling harness (bench_profile)
cmake -B build -DVECTORDB_PROFILE=ON
cmake --build build --target bench_profile
./build/bench/bench_profile --n 100000 --queries 5000 --ef 200
```

---

## Docs

| Document | Contents |
|----------|----------|
| [`docs/design.md`](docs/design.md) | System design overview — architecture, decisions, trade-offs |
| [`docs/api.md`](docs/api.md) | Python SDK + gRPC API reference; Phase 2 distributed design |
| [`docs/search_optimizations.md`](docs/search_optimizations.md) | SIMD optimization log with before/after measurements |
| [`docs/benchmarks.md`](docs/benchmarks.md) | SIFT-1M, GloVe-1.2M, memory profiling, Day 28 results |
| [`docs/checkpoint_recovery.md`](docs/checkpoint_recovery.md) | Atomic snapshot protocol and SIGKILL fault-injection matrix |
| [`docs/storage_modes.md`](docs/storage_modes.md) | Performance/compact memory layout and measured trade-offs |
