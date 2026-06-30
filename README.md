# VortexDB

A vector database built from scratch in C++ and Python. Built to understand how production systems like Pinecone, Weaviate, and Qdrant work under the hood.

**Core stack:** HNSW index · SIMD distance (AVX2 / NEON, auto-detected) · write-ahead log · metadata filtering · Python SDK · gRPC interface design

**SIFT-1M result:** 1,148 QPS at 99.2% Recall@1 (ef=200, M=16, single-threaded Python SDK).

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Python SDK (client.py)               │
│  open() · insert() · search() · remove() · checkpoint() │
└───────────────────────┬─────────────────────────────────┘
                        │ pybind11
┌───────────────────────▼─────────────────────────────────┐
│                    Engine  (server/engine.h)              │
│  shared_mutex — search holds read lock, writes exclusive │
│                                                          │
│  INSERT path:                                            │
│    1. append WAL record + fdatasync()  (crash safety)    │
│    2. insert into HnswIndex            (memory)          │
│    3. write vector to VectorFile       (mmap, O(1))      │
│    4. update MetadataIndex             (memory)          │
│                                                          │
│  SEARCH path:                                            │
│    1. HNSW beam search  (ef_search candidates)           │
│    2. post-filter by metadata allowlist                  │
│    3. return top-k sorted by distance                    │
│                                                          │
│  RECOVERY path (startup):                                │
│    1. load graph.bin + metadata.bin  (last checkpoint)   │
│    2. replay WAL records since checkpoint LSN            │
└──────┬──────────┬────────────┬───────────────┬──────────┘
       │          │            │               │
  ┌────▼───┐ ┌───▼────┐ ┌─────▼──────┐ ┌─────▼──────────┐
  │HnswIndex│ │VectorFile│ │    WAL     │ │ MetadataIndex  │
  │         │ │          │ │            │ │                │
  │ HNSW    │ │mmap flat │ │ append-    │ │ inverted index │
  │ graph   │ │ float32  │ │ only log   │ │ (string eq)    │
  │ + SIMD  │ │ array    │ │ fdatasync  │ │ + sorted array │
  │ distance│ │          │ │            │ │ (numeric range)│
  └────┬────┘ └────┬─────┘ └─────┬──────┘ └──────┬────────┘
       │(checkpoint)│             │                │(checkpoint)
  ┌────▼────────────▼─────────────▼────────────────▼────────┐
  │                      Disk                                │
  │  graph.bin  vectors.vdb  wal.log  metadata.bin           │
  └──────────────────────────────────────────────────────────┘
```

### Key design decisions

**HNSW** — probabilistic skip-list graph. O(log N) search, tunable recall via `ef_search`. Industry standard: hnswlib, faiss, and all major cloud vector databases use it.

**Flat arrays instead of pointer-based structures** — adjacency lists stored as a flat CSR array; vectors stored contiguously. Eliminates pointer chasing, enables hardware prefetch, reduces cache misses ~60% vs `std::vector<std::vector<float>>`.

**Write-ahead log** — every insert/delete appended to `wal.log` before touching memory. One `fdatasync()` per operation bounds the durability window. Recovery replays WAL since the last checkpoint, which bounds recovery time.

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

### SIFT-1M (1M vectors, dim=128, L2)

| ef_search | QPS | Recall@1 | Recall@10 | Recall@100 |
|----------:|----:|---------:|----------:|-----------:|
| 50  | 1,909 | 98.6% | 98.6% | 93.3% |
| 100 | 1,923 | 98.6% | 98.6% | 93.3% |
| **200** | **1,148** | **99.2%** | **99.7%** | **98.2%** |
| 400 |   678 | 99.2% | 99.9% | 99.6% |
| 800 |   390 | 99.3% | 99.9% | 99.9% |

Insert throughput: **177 vec/s** (single-threaded, ARM NEON, Python SDK). hnswlib achieves ~3,000–8,000 QPS; the gap is Python pybind11 overhead per query, not the C++ graph itself.

### GloVe-1.2M (1.18M vectors, dim=200, cosine)

| ef_search | QPS | Recall@1 |
|----------:|----:|---------:|
| 200 | 698 | 85.0% |
| 800 | 221 | 92.7% |

Lower recall than SIFT is expected — GloVe has known hubness and cosine-space topology issues. hnswlib achieves similar numbers.

### Memory overhead (dim=128, 200K vectors)

| | Value |
|-|-------|
| Raw vector storage | 512 B/vec |
| HNSW graph overhead | 661 B/vec |
| **Total** | **1,173 B/vec (2.3× raw)** |
| Peak RSS | 262 MB |

### SIMD optimization — search-phase breakdown

Five optimizations applied (flat visited array, flat vector store, flat node array, flat CSR adjacency, look-ahead prefetch). Net result: **1,669 → 4,407 QPS (+164%)**.

| Component | Before | After |
|-----------|-------:|------:|
| `unordered_set` visited tracking | 60.6% of search time | eliminated |
| `NeonL2::compute` (ARM NEON) | 12.4% | **51%** |
| `priority_queue` heap | 4.0% | 10% |

Distance compute went from 12% → 51% not because it got slower, but because the other overheads shrank. On x86 AVX2 (CI profiling): distance is 61% (AVX2 does 8 floats/cycle vs NEON's 4, finishing faster relative to other work).

See [`docs/search_optimizations.md`](docs/search_optimizations.md) for step-by-step optimization log.  
See [`docs/benchmarks.md`](docs/benchmarks.md) for recovery, stress test, and filtered search results.

---

## Python SDK

| Method | Description |
|--------|-------------|
| `vectordb.open(data_dir)` | Open or create a database |
| `db.create_collection(name, dimension, metric="l2")` | Create collection. `metric`: `"l2"` \| `"cosine"` \| `"ip"` |
| `db.drop_collection(name)` | Drop collection and delete files |
| `db.list_collections()` | List all collections |
| `db.insert(col, *, ids, vectors, metadata=None)` | Insert one or many vectors |
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
