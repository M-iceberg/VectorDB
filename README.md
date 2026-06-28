# VortexDB

A vector database built from scratch in C++. Supports insert, k-NN search, delete, and metadata-filtered search using an HNSW index with SIMD-accelerated distance functions, a write-ahead log for crash safety, and a Python SDK.

## Key features

- **HNSW index** — sub-linear approximate nearest-neighbor search
- **SIMD distance** — AVX2 on x86, NEON on ARM; auto-detected at compile time
- **WAL + checkpoint** — crash-safe writes; full recovery on restart
- **Metadata filtering** — insert string/numeric fields, filter at search time
- **Python SDK** — zero-copy numpy arrays, string or int IDs, batch insert

## Install (Python)

```bash
pip install . --no-build-isolation
```

Requires: cmake ≥ 3.20, a C++17 compiler, numpy, scikit-build-core, pybind11.

## Quickstart

```python
import vectordb
import numpy as np

# Open (or create) a database
db = vectordb.open("/tmp/demo")
db.create_collection("docs", dimension=128, metric="l2")

# Batch insert — zero-copy numpy float32
ids  = ["doc-a", "doc-b", "doc-c"]
vecs = np.random.rand(3, 128).astype(np.float32)
db.insert("docs", ids=ids, vectors=vecs)

# Search — returns original IDs
results = db.search("docs", query=vecs[0], top_k=2)
print(results)
# [{'id': 'doc-a', 'distance': 0.0}, {'id': 'doc-c', 'distance': 14.3}]
```

## Metadata and filtered search

```python
db.insert("docs", ids=["p1", "p2", "p3"], vectors=vecs,
          metadata=[
              {"category": "ml",  "year": 2024},
              {"category": "db",  "year": 2023},
              {"category": "ml",  "year": 2022},
          ])

# String equality filter
results = db.search("docs", query=vecs[0], top_k=10,
                    filters={"category": "ml"})

# Numeric range filter
results = db.search("docs", query=vecs[0], top_k=10,
                    filters={"year": {"$gte": 2023}})

# Combined (AND)
results = db.search("docs", query=vecs[0], top_k=10,
                    filters={"category": "ml", "year": {"$gte": 2023}})
```

Supported filter operators: `$gte` (≥), `$lte` (≤). Multiple keys are ANDed.

## Delete

```python
db.remove("docs", "doc-a")
```

Deleted IDs never appear in search results.

## Checkpoint

```python
db.checkpoint("docs")
```

Snapshots the index and metadata to disk, then truncates the WAL. The database recovers to a consistent state on restart even after a crash.

## Python SDK reference

| Method | Description |
|--------|-------------|
| `vectordb.open(data_dir)` | Open or create a database |
| `db.create_collection(name, dimension, metric="l2")` | Create a collection. metric: `"l2"` \| `"cosine"` \| `"ip"` |
| `db.drop_collection(name)` | Drop a collection and delete its files |
| `db.list_collections()` | List all collections |
| `db.insert(col, *, ids, vectors, metadata=None)` | Insert one or many vectors |
| `db.remove(col, id)` | Soft-delete a vector |
| `db.search(col, *, query, top_k=10, ef_search=64, filters=None)` | ANN search |
| `db.checkpoint(col)` | Flush snapshot to disk |

## Build from source (C++ only)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run tests
./build/tests/unit/test_smoke
./build/tests/unit/test_hnsw_ut
./build/tests/unit/test_engine_recovery_int
# ... (see .github/workflows/ci.yml for full list)
```

## Architecture

```
vectordb.open()
    └── VectorDB (Python wrapper — client.py)
            └── Engine (C++ — server/engine.h)
                    ├── HnswIndex   — HNSW graph, SIMD distance
                    ├── VectorFile  — mmap-backed flat vector store
                    ├── Wal         — append-only write-ahead log
                    └── MetadataIndex — inverted index for filtered search
```

On every `insert`: vector written to VectorFile, WAL record appended, node added to HnswIndex, metadata updated in MetadataIndex.

On `search`: HNSW beam search with inflated `ef_search` → post-filter by metadata allowlist → return top-k.

On restart: load latest checkpoint snapshot, then replay WAL records written after the checkpoint.
