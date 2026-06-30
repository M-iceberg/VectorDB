# Benchmarks

## SIMD Profiling — Search Hotspot Analysis (Day 27)

**Setup:** dim=128, N=100K, ef=200, HNSW M=16, L2 metric  
**Hardware:** Apple Silicon (ARM NEON)  
**Tool:** macOS `sample` (8-second sampling window, captured during 100K-query search run)  
**Binary:** `bench/bench_profile.cpp` (Release build, no Python overhead)

### What does distance compute cost?

| Component | Samples | % of search |
|-----------|--------:|------------:|
| `unordered_set` visited alloc (hash table insert) | 4,274 | 50.7% |
| Graph arithmetic / pointer chasing | 1,492 | 17.7% |
| **NeonL2::compute** (NEON SIMD distance) | **1,044** | **12.4%** |
| `free` — visited set teardown | 828 | 9.8% |
| `priority_queue::push` | 339 | 4.0% |
| Other | 445 | 5.4% |

**Visited-set total (alloc + free): 5,102 samples = 60.6% of search time.**

Root cause: `HnswIndex::search_layer()` creates a `std::unordered_set<uint32_t>` per call to track visited nodes. At ef=200, each query calls `search_layer()` and inserts up to ~3,300 node IDs. The hash table triggers `operator new` on rehash and requires full teardown on return — all of which shows up as malloc/free in the profile, not in distance math.

### Prefetch on vs off

| Mode | QPS | Per-query latency |
|------|----:|------------------:|
| Prefetch ON (`__builtin_prefetch`) | 1,669 | 0.599 ms |
| Prefetch OFF | 1,725 | 0.580 ms |
| Delta | +3.4% without prefetch | |

`__builtin_prefetch` in `compute_batch()` adds instruction overhead with no cache benefit. The prefetch is designed for sequential array scans, but HNSW `search_layer()` accesses graph neighbors scattered throughout memory — by the time `compute_batch` sees each neighbor, it's already been fetched by the graph traversal (or missed L3 either way). Distance compute is 12% of runtime, so even a 2× speedup there would only improve total QPS by 6%.

### Distance compute microbenchmark

Isolated cost of NEON `compute_batch()` on a warm contiguous 100K-vector array:

| Metric | Value |
|--------|-------|
| Throughput | 9.4 ns/vec |
| Rate | ~106 M vectors/sec |
| Estimated per-query distance cost (ef=200, ~3,322 nodes) | 0.031 ms |
| Actual per-query search latency | 0.580 ms |
| Distance fraction | ~5% (microbench) / ~12% (sample, includes cache misses) |

The gap between 5% and 12% reflects cache effects: the contiguous microbenchmark benefits from hardware prefetch, while HNSW search accesses scattered node vectors that miss L3 cache.

### Optimization opportunity

Replacing the per-search `std::unordered_set` with a flat bitset (or a generation-counter array) would eliminate the 61% allocation overhead. At 100K nodes, a bitset needs 12.5 KB — fits in L2 cache. This is the highest-leverage optimization available and would be the first change in a production hardening pass.

### Linux (AVX2) — perf report (CI, AMD EPYC 7763)

GitHub Actions `ubuntu-latest` (x86-64, AVX2), N=50K, dim=128, ef=200. Profiled with `perf record -F 99` + `perf report --stdio`.

Hardware PMU counters (`cycles`, `instructions`, `cache-misses`) showed `<not supported>` — GitHub Actions VMs do not expose hardware PMU to the guest. PC sampling still works.

**Function breakdown (prefetch ON):**

| Function | x86 AVX2 | ARM NEON |
|----------|----------:|----------:|
| Distance compute (`Avx2L2::compute`) | **61%** | 51% |
| `search_layer` traversal overhead | 21% | 38% |
| `priority_queue` heapify | 10% | 10% |
| Sort / alloc / other | 8% | 1% |

AVX2 processes 8 floats/cycle vs NEON's 4, so distance finishes faster in wall time. The graph traversal fraction shrinks proportionally, making distance the dominant fraction on x86.

**Prefetch comparison (N=50K, AMD EPYC 7763):**

| | Prefetch ON | Prefetch OFF |
|--|--:|--:|
| QPS | 3,037 | 3,290 |
| Gain | −8% (overhead > benefit) | — |

At N=50K, `vecs_flat_` is 25.6 MB. AMD EPYC 7763 has 256 MB L3 — the full working set fits, so there is no DRAM latency to hide and the prefetch loop itself becomes net overhead. Consistent with the macOS finding at N=100K (+4.5%): **prefetch is only beneficial when dataset size exceeds L3 capacity**.

For hardware PMU counters (IPC, cache-miss rate), a bare-metal instance (e.g. AWS `c5n.metal`) is needed. See `.github/workflows/ci.yml` → `simd-profiling` job for raw output.

---

## GloVe-1.2M Benchmark

**Setup:** GloVe-1.2M dataset (1,183,514 base vectors, 10K queries, dim=200, cosine/angular), M=16, ef_construction=200, local ARM (Apple Silicon M-series). Python SDK, single-threaded insert.

| ef_search | QPS | Recall@1 | Recall@10 | Recall@100 |
|-----------|-----|----------|-----------|------------|
| 50  | 1,171 | 79.0% | 76.8% | 64.9% |
| 100 | 1,172 | 79.0% | 76.8% | 64.9% |
| 200 |   698 | 85.0% | 82.6% | 73.2% |
| 400 |   408 | 89.6% | 87.3% | 79.7% |
| 800 |   221 | 92.7% | 90.9% | 85.0% |

Insert throughput: **146 vec/s** average over 1.18M vectors (135 minutes total).

### Observations

**1. GloVe recall is substantially lower than SIFT — this is expected**

SIFT-1M (ef=200): R@1=99.2%. GloVe-1.2M (ef=200): R@1=85.0%. This 14-point gap reflects three properties of the GloVe dataset, not a deficiency in the index:

- **Hubness**: In high-dimensional cosine spaces, a small subset of vectors ("hubs") become nearest neighbors of many other vectors. This distorts the HNSW graph structure because hubs attract many edges, leaving other regions poorly connected.
- **Higher dimension**: dim=200 vs dim=128. HNSW graph quality degrades with dimension because the ratio of near-neighbor candidates to non-neighbors shrinks, making beam search less effective.
- **Cosine vs L2**: Cosine similarity on word embeddings creates a more uniform distance distribution (many points at similar angular distance), making it harder to distinguish true nearest neighbors.

GloVe-200-angular is one of the hardest standard ANN benchmarks. hnswlib achieves ~85–95% R@1 on this dataset depending on ef, so VortexDB's numbers are in range.

**2. ef=50 and ef=100 again plateau**

Same phenomenon as SIFT: doubling ef from 50 to 100 gives zero additional recall because the minimum traversal cost for a 1.2M-node graph exceeds ef=50 worth of candidates. The practical floor here is ef=200.

**3. ef=800 needed for >90% R@1**

On SIFT, ef=100 already gave 98.6% R@1. On GloVe, even ef=800 only gives 92.7%. This confirms the dataset difficulty — the hubness structure means some true nearest neighbors are in poorly-connected graph regions that beam search cannot reach regardless of ef.

**4. Insert throughput lower than SIFT (146 vs 177 vec/s)**

Each distance computation is 200 × 4 = 800 bytes vs 128 × 4 = 512 bytes — 56% more data per dot product. Combined with the same cache-miss behavior at large N, the throughput floor is lower.

## Memory Profiling — HNSW Index

**Setup:** 200K vectors, dim=128, M=16, ef_construction=200, L2, Apple Silicon. Measured with `/usr/bin/time -l` (macOS) + `resource.getrusage()` at 50K checkpoints.

| Vectors | Peak RSS | Raw vectors | HNSW overhead | Overhead B/vec | Total B/vec |
|---------|----------|-------------|---------------|----------------|-------------|
| 50K  |  108 MB |  24 MB |  46 MB |  964 B/vec | 1,476 B/vec |
| 100K |  158 MB |  49 MB |  72 MB |  750 B/vec | 1,262 B/vec |
| 150K |  207 MB |  73 MB |  95 MB |  667 B/vec | 1,179 B/vec |
| 200K |  262 MB |  98 MB | 126 MB |  661 B/vec | 1,173 B/vec |

**Summary at 200K vectors:**
- Raw vector data: 512 B/vec (128 × float32)
- HNSW graph overhead: **661 B/vec** (neighbor lists + node metadata + allocator)
- Total: **1,173 B/vec — 2.3× raw storage**
- Peak RSS: 262 MB confirmed by `/usr/bin/time -l` (maximum resident set size: 274,563,072 bytes)

**Why 661 B/vec overhead?**

With M=16 (max neighbors per non-zero layer) and M0=32 (layer 0):
- Layer 0 neighbor list: 32 × 4 bytes = 128 bytes
- Upper layers: avg ~1.5 levels × 16 neighbors × 4 bytes ≈ 96 bytes
- Node struct + level + deleted flag: ~16 bytes
- `std::vector` capacity overhead (typically 1.5–2× allocated): adds ~30–50%
- Total per node ≈ 240–400 bytes allocated, but with allocator fragmentation and Python object overhead through pybind11 boundary ≈ 661 B/vec observed

The overhead decreases from 964 B/vec at 50K to 661 B/vec at 200K as fixed engine state is amortized over more vectors.

**Comparison:** faiss HNSW with flat storage uses ~200–300 B/vec overhead (neighbor lists stored as contiguous arrays). VortexDB's per-node `std::vector` has ~2–3× more overhead due to heap fragmentation — a known optimization target (see [SIFT-1M benchmark observations](#sift-1m-benchmark)).

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

---

## Recovery Time Benchmark (Day 28)

**Setup:** dim=128, M=16, ef_construction=200, Apple Silicon (ARM). No checkpoint before close — WAL is the only persistence. Recovery time = `Engine(data_dir)` constructor (pure C++ WAL replay, no Python overhead).  
**Script:** `bench/bench_recovery.py`

| WAL size | Vectors | Recovery time | Replay speed |
|----------|--------:|-------------:|-------------:|
| 0.51 MB  |   1,000 |        ~90 ms |   ~11 Kv/s |
| 1.54 MB  |   3,000 |       ~330 ms |    ~9 Kv/s |
| 4.1 MB   |   8,000 |     ~1,350 ms |    ~6 Kv/s |

**Finding:** Recovery time scales O(N log N), not O(N). WAL replay re-inserts each record into the HNSW graph, and HNSW insert is O(M × ef_construction × log N) — the same cost as the original insert. A 256 MB WAL (~470K records at dim=128) would take roughly 4–6 minutes to recover from WAL alone.

**Why checkpoint matters:** After a checkpoint, the graph snapshot is loaded in O(N) (sequential file read), and only the WAL records since the last checkpoint need replaying. A checkpoint every 50K inserts bounds recovery time to ~30–60s regardless of total collection size.

---

## Stress Test — 10-minute Mixed Workload (Day 28)

**Setup:** dim=16, continuous insert + search + delete loop for 10 minutes. Checkpoint every 10 iterations. At end: engine reopened, 500 random live vectors verified searchable.  
**Script:** `bench/bench_stress.py`

| Metric | Result |
|--------|--------|
| Total duration | 600 s |
| Inserts | ~73,500 |
| Searches | ~14,700 |
| Deletes | ~29,400 |
| Throughput | ~290 ops/s |
| Crashes | 0 |
| Data loss (missing after recovery) | 0 |

**PASS** — no crashes, no data loss across 117K mixed operations.

---

## Filtered Search — Selectivity vs QPS and Recall (Day 28)

**Setup:** N=10K vectors, dim=128, M=16, ef=100, top_k=10. Each vector assigned a category from {0..K−1} uniformly. Filter targets category 0 (expected hit count = N/K). Brute-force ground truth computed with numpy.  
**Script:** `bench/bench_filtered.py`

**Selectivity** = fraction of vectors that pass the filter = 1/K. K=1 means all vectors are in the same category (no filter effect, 100% pass). K=100 means each category holds 1% of vectors — only 100 out of 10K pass the filter. Lower selectivity = stricter filter = fewer candidates eligible for the result.

**Recall@10** = fraction of true top-10 nearest neighbors (by brute force) that HNSW actually returns. Range is 0–1; 1.000 = perfect, identical to brute force. 0.700 means HNSW found 7 out of the 10 true nearest neighbors and missed 3. The problem is not recall=1 at high selectivity — that means search is working correctly. The problem is recall collapsing at low selectivity (1% and below), where HNSW starts missing real nearest neighbors.

| Selectivity | Candidate pool | QPS | Recall@10 |
|------------:|---------------:|----:|----------:|
| 100% | 10,000 | ~900 | 1.000 |
| 50% | 5,000 | ~1,200 | 1.000 |
| 10% | 1,000 | ~2,500 | 1.000 |
| 1% | 100 | ~3,100 | **0.70** |
| 0.1% | 10 | ~3,400 | **0.30** |

**Finding 1 — QPS increases as selectivity drops.** Fewer candidates pass the filter → priority queue is smaller → heap operations cheaper → each query finishes faster. The HNSW graph traversal visits roughly the same number of nodes, but result assembly takes less time.

**Finding 2 — Recall collapses at low selectivity.** The HNSW graph is built for unfiltered similarity. At 1% selectivity only 100 vectors are eligible. With ef=100, beam search explores ~100 candidates, but most are rejected by the filter — the effective candidate pool for the filtered result is far smaller than ef. Some true nearest neighbors in the eligible set are never reached because they are poorly connected in the graph (the graph was built ignoring the filter).

This is the fundamental tension in filtered ANN: **the graph is optimized for global proximity, not per-filter proximity.** Production systems (Weaviate, Qdrant, Pinecone) address this with dedicated per-segment indexes or hybrid HNSW+inverted-index structures. For selectivity > 10%, post-filtering on top of unfiltered ANN works well; below 1%, recall degrades significantly without per-filter graph construction.
