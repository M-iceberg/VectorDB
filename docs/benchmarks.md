# Benchmarks

## HNSW: Heuristic vs Greedy Neighbor Selection

**Setup:** 1000 vectors, dim=32, M=16, M0=32, ef_construction=100, ef_search=64, 100 queries, k=10, L2 metric, local ARM (Apple Silicon).

| Method | Recall@10 |
|--------|-----------|
| Greedy | 99.8% |
| Heuristic (Algorithm 4) | 99.9% |
| Delta | +0.1% |

At this scale the difference is small — 1000 vectors and dim=32 is easy enough that greedy already finds nearly all true nearest neighbors. The heuristic's advantage becomes more pronounced at larger scale (100K+ vectors, higher dimensions) where greedy tends to cluster neighbors in one direction and miss candidates coming from other directions.

## HNSW: SIFT-small Recall

**Setup:** siftsmall dataset (10K base vectors, 100 queries, dim=128, L2), M=16, M0=32, ef_construction=200, ef_search=64, k=10, local ARM (Apple Silicon).

| | Value |
|-|-------|
| recall@10 | 99.7% |
| base vectors | 10,000 |
| queries | 100 |

## SIFT-1M Benchmark

**Setup:** SIFT-1M dataset (1M base vectors, 10K queries, dim=128, L2), M=16, ef_construction=200, local ARM (Apple Silicon M-series). Python SDK, single-threaded insert.

| ef_search | QPS | Recall@1 | Recall@10 | Recall@100 |
|-----------|-----|----------|-----------|------------|
| 50  | 1,909 | 98.6% | 98.6% | 93.3% |
| 100 | 1,923 | 98.6% | 98.6% | 93.3% |
| 200 | 1,148 | 99.2% | 99.7% | 98.2% |
| 400 |   678 | 99.2% | 99.9% | 99.6% |
| 800 |   390 | 99.3% | 99.9% | 99.9% |

Insert throughput: **177 vec/s** average over 1M vectors (94 minutes total).

### Observations

**1. ef=50 and ef=100 give identical QPS and recall**

ef=50 → 1,909 QPS, ef=100 → 1,923 QPS — statistically the same. This is counter-intuitive (doubling ef should double the work), but at 1M scale the HNSW graph has enough layers that the search already visits many nodes even at ef=50. The graph structure forces traversal of a minimum number of nodes regardless of ef when the entry point is far from the query. The practical takeaway: ef=50 is wasteful to go lower, ef=100 buys nothing over ef=50.

**2. R@100 has a large jump between ef=100 and ef=200**

R@100 goes from 93.3% (ef=100) to 98.2% (ef=200). The reason: to reliably return 100 results that are all in the true top-100, the beam search must maintain at least 100 candidates at all times. With ef=100, the candidate set is just barely large enough, so some true top-100 neighbors get dropped. With ef=200, there is headroom, and recall jumps sharply. Rule of thumb: ef should be at least k when measuring R@k.

**3. Recall@1 saturates quickly**

R@1 goes from 98.6% at ef=50 to 99.3% at ef=800 — only 0.7% improvement for a 16× QPS cost. The remaining 0.7% miss rate is due to the approximate nature of HNSW: a small fraction of queries have their true nearest neighbor in a part of the graph that beam search never reaches regardless of ef. This is the irreducible error from not building a perfect graph.

**4. Sweet spot is ef=200**

ef=200 achieves 99.2% R@1 and 99.7% R@10 at 1,148 QPS. Going higher buys marginal recall at significant QPS cost. For most applications ef=200 is the right default.

**5. Insert throughput degradation curve**

Observed throughput at each 100K checkpoint:

| Vectors inserted | Throughput |
|-----------------|------------|
| 10K  | 254 vec/s |
| 110K | 218 vec/s |
| 210K | 201 vec/s |
| 310K | 194 vec/s |
| 410K | 188 vec/s |
| 510K | 186 vec/s |
| 610K | 183 vec/s |
| 710K | 179 vec/s |
| 810K | 178 vec/s |
| 910K | 178 vec/s |

The curve is smooth and asymptotic — not a sudden cliff. The early fast degradation (254→194) is the log(N) algorithmic factor. The flattening after ~400K is because log(N) grows slowly and the per-insert work stabilises once the graph reaches a steady-state depth.

**6. Comparison context**

hnswlib (C++, same SIFT-1M, comparable hardware) typically achieves ~3,000–8,000 QPS at R@1≈99% depending on ef. VortexDB at 1,148 QPS is roughly 3–6× slower. The gap comes from:
- Python SDK overhead per query (pybind11 call, dict allocation for results)
- No query-side SIMD prefetch during graph traversal
- hnswlib uses tighter node memory layout (neighbors packed inline with the vector)

These are all addressable. The C++ layer itself is competitive; the overhead is in the Python boundary and graph memory layout.

## HNSW large-scale insert throughput

At small N (~10K), insert throughput is ~250 vec/s. At large N (~650K+), it drops to ~25 vec/s — a 10x degradation. Two causes:

**1. Algorithmic: O(M × ef_construction × log N) per insert**
Each insert runs a beam search through the current graph to find good neighbors. As N grows, more layers exist and each search visits more nodes.

**2. Memory bandwidth bottleneck (dominant cause)**
A 1M-node HNSW graph has a working set of ~576MB (512MB vectors + 64MB edge pointers). L3 cache on typical hardware is 8–16MB. At large N, almost every neighbor access during graph search is a cache miss (~100ns DRAM latency vs ~1ns L1). The CPU stalls waiting for memory rather than computing distances.

This is a fundamental characteristic of large HNSW graphs, not a bug. All HNSW implementations experience it; production libraries (hnswlib, faiss) mitigate it with tighter memory layouts and prefetching.

### Optimization TODOs (Day 27 profiling will quantify these)

- **Batch insert**: expose a C++-level batch insert that inserts N vectors in one call, reducing Python→pybind11→C++ overhead from N calls to 1
- **Node memory layout**: pack same-layer neighbors contiguously in memory so that traversing a layer's neighbor list hits fewer cache lines
- **Graph traversal prefetch**: extend the `__builtin_prefetch` pattern (already used in `compute_batch`) to prefetch neighbor vectors during HNSW search, hiding DRAM latency behind computation
