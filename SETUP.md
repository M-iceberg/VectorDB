# VectorDB — Project Context for Claude

## What this project is

A vector database built from scratch in C++. Supports insert, k-NN search, and delete
using an HNSW (Hierarchical Navigable Small World) index with SIMD-accelerated distance
functions. The goal is a working, production-quality vector DB by Day 30.

GitHub repo: https://github.com/M-iceberg/VectorDB

---

## Progress

### Completed

**Day 1–2: Scalar distance functions**
- `src/core/distance.h` — `DistanceCompute` base class, `Metric` enum (L2/Cosine/InnerProduct)
- `src/core/distance_naive.cpp` — scalar implementations (NaiveL2, NaiveCosine, NaiveIP)
- `create()`, `create_scalar()`, `create_avx2()` factory methods
- `compute_batch()` with `__builtin_prefetch` in base class

**Day 3: ARM NEON SIMD**
- `src/core/distance_neon.cpp` — NEON implementations (4-wide float32x4_t)
- Overrides `create()` on ARM
- `tests/unit/test_distance_neon.cpp` — 25 tests, NEON vs scalar correctness

**Day 4: x86 AVX2 SIMD**
- `src/core/distance_avx2.cpp` — AVX2 implementations (8-wide __m256), runtime dispatch
- `__builtin_cpu_supports("avx2")` in `create()` for runtime fallback to scalar
- `tests/unit/test_distance_avx2.cpp` — 32 tests

**Day 5: Benchmarks + CI**
- `bench/bench_distance.cpp` — Google Benchmark: scalar vs NEON vs AVX2, prefetch effect,
  scalar tail, unaligned memory
- `.github/workflows/ci.yml` — ARM + x86 CI (push/PR to main)
- `.github/workflows/bench.yml` — manual benchmark trigger, commits JSON results
- `bench_results/` — benchmark summaries and raw JSON from CI

**Day 6: HNSW index**
- `src/core/hnsw_node.h` — `HnswNode` struct (id, layer, neighbors, tombstone)
- `src/core/hnsw_index.h` — `HnswConfig`, `HnswIndex` API (insert/search/remove/size)
- `src/core/hnsw_index.cpp` — full HNSW implementation:
  - `assign_layer()` — exponential layer distribution (ml = 1/ln(M))
  - `search_layer()` — beam search with min-heap candidates + max-heap results
  - `select_neighbors()` — greedy: take M closest from candidates
  - `add_edge()` — bidirectional edges with pruning (evict farthest if over M_max)
  - `insert()` — greedy descent to find entry point, then beam search + edge building
  - `search()` — greedy descent + layer-0 beam search, filter tombstones
  - `remove()` — soft delete (tombstone only, no relinking)
- `tests/unit/test_hnsw.cpp` — 19 tests: basic, correctness, tombstone, recall ≥ 90%,
  all three metrics, ef_search effect, size accuracy

### Not yet committed
Day 6 changes (hnsw_index.cpp, test_hnsw.cpp, updated comments in hnsw_node.h /
hnsw_index.h) are complete locally but not yet pushed. Run `git status` to confirm,
then commit and push.

---

## What's next

### Day 7–10: HNSW improvements
- Diversity heuristic for `select_neighbors` (Algorithm 4 from the paper)
- Persistent graph: save/load HNSW graph to disk (`src/storage/graph_serializer.h`)
- `bench/bench_hnsw.cpp` — benchmark insert throughput and search recall vs latency

### Day 11–12: Storage layer
- `src/storage/vector_file.cpp` — implement mmap-backed flat vector storage
  (header already complete: `VectorFile::append`, `read`, `slot_count`)
- `src/storage/wal.cpp` — implement append-only write-ahead log
  (header already complete: `Wal::append`, `sync`, `iterate`, `truncate_before`)
- Replace `unordered_map<NodeId, vector<float>>` in HnswIndex::Impl with VectorFile

### Day 13: Arena allocator
- `src/core/arena_allocator.cpp` — implement slab arena (header already complete)

### Day 14+: Engine, gRPC server, metadata index, checkpointing, client

---

## Build and test

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build all
cmake --build build

# Run tests
./build/tests/unit/test_smoke
./build/tests/unit/test_distance
./build/tests/unit/test_hnsw
# ARM only:
./build/tests/unit/test_distance_neon
# x86 only:
./build/tests/unit/test_distance_avx2

# Build benchmarks
cmake -B build -DCMAKE_BUILD_TYPE=Release -DVECTORDB_BENCH=ON
cmake --build build --target bench_distance
./build/bench/bench_distance
```

---

## Code conventions

- All code in `namespace vectordb`
- Pimpl pattern for all public classes (`struct Impl; std::unique_ptr<Impl> impl_`)
- Platform SIMD guard: `#if defined(VECTORDB_ARCH_ARM)` / `#if defined(VECTORDB_ARCH_X86)`
  (defined in `cmake/SimdDetect.cmake`)
- Anonymous namespace for all internal helpers in `.cpp` files
- `create()` is the platform-dispatching factory; `create_scalar()` always returns scalar
- No Co-Authored-By lines in commits
- Commit messages: no "Day X:" prefix, just describe what was implemented

---

## Key files

| File | Status | Notes |
|------|--------|-------|
| `src/core/distance.h` | complete | base class + factory declarations |
| `src/core/distance_naive.cpp` | complete | scalar + create_scalar + compute_batch |
| `src/core/distance_neon.cpp` | complete | ARM NEON, overrides create() |
| `src/core/distance_avx2.cpp` | complete | x86 AVX2, runtime dispatch in create() |
| `src/core/hnsw_node.h` | complete | node data structure |
| `src/core/hnsw_index.h` | complete | public API + HnswConfig |
| `src/core/hnsw_index.cpp` | complete | full HNSW algorithm |
| `src/core/arena_allocator.h` | complete | header only, impl is Day 13 |
| `src/core/arena_allocator.cpp` | stub | to be implemented Day 13 |
| `src/storage/vector_file.h` | complete | header only, impl pending |
| `src/storage/vector_file.cpp` | stub | to be implemented Day 11 |
| `src/storage/wal.h` | complete | header only, impl pending |
| `src/storage/wal.cpp` | stub | to be implemented Day 12 |
| `src/storage/graph_serializer.h` | complete | header only, impl pending Day 7 |
| `src/server/engine.h` | complete | header only, impl pending |
| `bench/bench_distance.cpp` | complete | full benchmark suite |
| `bench/bench_hnsw.cpp` | stub | to be implemented Day 7–10 |
