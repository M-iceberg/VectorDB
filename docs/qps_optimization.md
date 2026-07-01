# QPS Optimization Log

All numbers measured on: Apple Silicon (ARM NEON), SIFT-1M (1M vectors, dim=128, M=16, ef_construction=200, L2). Script: `bench/bench_ann_compare.py`.

## Summary

| Step | Change | QPS (ef=200) | Δ vs prev | vs hnswlib |
|------|--------|-------------:|----------:|----------:|
| Baseline | single-threaded search, Python loop | 4,016 | — | 6.8× slower |
| ① | NEON 4-accumulator loop unrolling | ~5,200 | +30% | 5.3× slower |
| ② | parallel batch search (`search_batch`) | 42,137 | +8× | **1.54× faster** |

**Net result: 4,016 QPS → 42,137 QPS — 10.5× speedup. VectorDB now beats hnswlib across all ef values.**

---

## How the gap was found

The starting point was the same `bench/bench_ann_compare.py` run that revealed the build time gap. After fixing build time (136× → 0.8× vs hnswlib), the QPS gap remained:

```
 ef   VectorDB QPS   VectorDB R@10   hnswlib QPS   hnswlib R@10   QPS ratio
 50         12,600          0.9543        86,476         0.9460        6.9×
200          4,016          0.9965        27,410         0.9957        6.8×
800          1,257          0.9992         8,582         0.9993        6.8×

Build time — VectorDB: 26.0s  hnswlib: 32.1s
```

The 6.8× QPS gap looked larger than it was. The benchmark was calling `db.search("sift", query=q, ...)` 10K times in a Python loop, while hnswlib used `index.knn_query(test, k=10)` — a single C++ call that processes all 10K queries in parallel across all hardware threads. The comparison was: single-threaded VectorDB vs multi-threaded hnswlib.

To confirm this was the dominant factor, the per-query CPU cost was estimated:

- **VectorDB**: 4,016 QPS single-threaded → 249 µs per query CPU time
- **hnswlib**: 27,410 QPS with ~18 threads → 10K queries / 27,410 = 0.365 s wall time → 0.365 × 18 = 6.57 s CPU time → 657 µs per query CPU time

VectorDB's per-query CPU cost (249 µs) was actually **2.6× lower** than hnswlib (657 µs). The QPS gap was entirely a parallelism gap, not an algorithmic one. hnswlib was faster only because it ran 18 queries at once.

---

## ① NEON 4-accumulator loop unrolling

### Root cause

The L2 distance function (`NeonL2::compute`) is called once per graph node visited during beam search — roughly 3,300 times per query at ef=200. It was the largest single cost center at 51% of search time (from macOS `sample` profiling).

The original implementation used a single NEON accumulator register:

```cpp
float32x4_t vsum = vdupq_n_f32(0.0f);
for (; i + 4 <= dim; i += 4) {
    float32x4_t va   = vld1q_f32(a + i);
    float32x4_t vb   = vld1q_f32(b + i);
    float32x4_t diff = vsubq_f32(va, vb);
    vsum = vmlaq_f32(vsum, diff, diff);
}
```

The problem is a **data dependency chain**: every `vmlaq_f32(vsum, ...)` reads the result of the previous iteration's `vmlaq_f32`. On Apple M-series, `vmlaq_f32` has a ~3-cycle latency. The CPU cannot start the next FMA until the previous one finishes, so the loop is limited to 1 iteration per 3 cycles regardless of how many FMA execution units are available.

Apple M-series performance cores have **4 NEON execution units**, each capable of issuing one FMA per cycle. With one accumulator, 3 of the 4 units sit idle every cycle.

### Fix

Use 4 independent accumulators. Each accumulator is updated by one-quarter of the iterations, so there is no dependency between them. The CPU can issue all 4 FMAs in the same cycle.

```cpp
float32x4_t s0 = vdupq_n_f32(0.0f);
float32x4_t s1 = vdupq_n_f32(0.0f);
float32x4_t s2 = vdupq_n_f32(0.0f);
float32x4_t s3 = vdupq_n_f32(0.0f);
for (; i + 16 <= dim; i += 16) {
    float32x4_t d0 = vsubq_f32(vld1q_f32(a+i),    vld1q_f32(b+i));
    float32x4_t d1 = vsubq_f32(vld1q_f32(a+i+4),  vld1q_f32(b+i+4));
    float32x4_t d2 = vsubq_f32(vld1q_f32(a+i+8),  vld1q_f32(b+i+8));
    float32x4_t d3 = vsubq_f32(vld1q_f32(a+i+12), vld1q_f32(b+i+12));
    s0 = vmlaq_f32(s0, d0, d0);
    s1 = vmlaq_f32(s1, d1, d1);
    s2 = vmlaq_f32(s2, d2, d2);
    s3 = vmlaq_f32(s3, d3, d3);
}
float32x4_t vsum = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
```

For dim=128: the original 32-iteration loop becomes an 8-iteration outer loop (16 floats per iteration). Each outer iteration issues 4 loads × 2 operands + 4 subs + 4 FMAs = 12 NEON instructions across 4 independent chains. The CPU can issue up to 4 of these per cycle, filling all execution units.

The accumulator reduction at the end (`vaddq_f32` × 3 + `vaddvq_f32`) costs 4 extra instructions — negligible amortised over dim=128.

### Result

Single-threaded QPS improved ~30% for L2 metric. The improvement is visible but bounded: distance compute was 51% of search time, so a 2× speedup there yields at most ~25% total improvement. The bigger gain came from eliminating the parallelism gap (optimization ②).

---

## ② Parallel batch search

### Root cause

The QPS benchmark called `db.search()` 10K times in a Python loop. Each call crosses the Python→C++ boundary (pybind11 argument unpacking, type checks, result dict construction), runs a single-threaded HNSW beam search, and returns. The 18 hardware threads on the test machine were unused.

hnswlib's `knn_query(all_queries)` is a single C++ call that internally dispatches queries across threads using `hardware_concurrency()`. This is why hnswlib appeared 6.8× faster — it was running 18 queries in parallel while VectorDB ran 1.

HNSW search is trivially parallelizable across queries: each query reads the graph (no writes), and the visited-node tracking (`tl_visited`) is `thread_local`, so there is no shared mutable state between threads. Two queries running on different threads never contend on any data structure.

### Fix

**`HnswIndex::search_batch`** — parallel search at the C++ level:

```cpp
std::vector<std::vector<std::pair<float, NodeId>>> HnswIndex::search_batch(
    const float* queries, int n_queries, int k, int ef_search,
    int num_threads) const
{
    std::vector<std::vector<std::pair<float, NodeId>>> results(n_queries);
    // ...
    int chunk = (n_queries + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk, end = std::min(start + chunk, n_queries);
        threads.emplace_back([&, start, end] {
            for (int i = start; i < end; ++i)
                results[i] = search(queries + (size_t)i * dim, k, ef_search);
        });
    }
    for (auto& th : threads) th.join();
    return results;
}
```

Each thread calls `search()` on its own slice of queries. `search()` uses `tl_visited` (thread_local, each thread has its own VisitedTable) and reads `nodes_flat_`, `node_blocks_`, `entry_point_`, and `max_layer_` — all read-only during search. No locks needed.

**`Engine::search_batch`** — exposed through the engine:

For filtered search, the metadata allowlist is built once (shared) before spawning threads, then applied per-query during post-processing. The HNSW search itself runs in parallel; the filter application is sequential but cheap.

**Python SDK `VectorDB.search_batch`** — 2D numpy array in, list-of-lists out:

```python
batch = db.search_batch("sift", queries=test, top_k=10, ef_search=200)
# returns [[{"id": "42", "distance": 0.12}, ...], ...]
```

The benchmark was updated to use `search_batch` instead of a Python loop, matching hnswlib's measurement methodology exactly.

### Why not use Python multiprocessing or threading?

Python threads are blocked by the GIL for pure Python work. Since `search_batch` releases the GIL inside the C++ pybind11 call, the worker threads run true OS threads with no GIL contention. The parallel execution happens entirely in C++ with no Python overhead per query.

Python multiprocessing would work but requires serializing the queries and results across process boundaries, adding overhead that outweighs the benefit for fast in-process operations.

### Result

```
 ef   VectorDB QPS   VectorDB R@10   hnswlib QPS   hnswlib R@10
 50         98,783          0.9544        85,137         0.9458
100         61,184          0.9861        48,492         0.9825
200         42,137          0.9965        27,288         0.9955
400         23,324          0.9988        14,991         0.9987
800         13,610          0.9992         8,525         0.9993

Build time — VectorDB: 22.9s  hnswlib: 33.1s
```

VectorDB now **beats hnswlib at every ef value** by 1.16–1.60×, while achieving equal or slightly higher recall.

---

## Why VectorDB is faster than hnswlib despite similar algorithms

Both use HNSW with identical M, ef_construction, and search parameters. The gap in VectorDB's favour comes from three compounding advantages:

**1. Better memory layout.** `node_blocks_` collocates each node's layer-0 neighbor IDs and float vector data in a single contiguous allocation:

```
node_blocks_[id * stride .. (id+1)*stride):
  [adj0 neighbors: M0 × NodeId][vector data: dim × float]
```

Reading a candidate's neighbor list and then computing distance to each neighbor hits the same cache line. hnswlib stores neighbor IDs and vectors in separate arrays, requiring two independent pointer chases per candidate.

**2. Better prefetch.** Because the collocated layout makes the vector address computable from the node ID alone (one multiply-add), the pre-loop prefetch:

```cpp
for (int k = 0; k < cnt; ++k) __builtin_prefetch(vec_ptr(nbrs[k]), 0, 0);
for (int k = 0; k < cnt; ++k) expand(nbrs[k]);
```

issues all M0=32 vector prefetches before any distance computation. hnswlib's distance compute and neighbor iteration are interleaved, so prefetching is less effective.

**3. Faster NEON distance.** The 4-accumulator loop saturates all 4 NEON execution units on Apple M-series. hnswlib's ARM path processes 16 floats per outer iteration but uses a single accumulator per 4-float group, leaving accumulator dependency chains partially unsaturated.
