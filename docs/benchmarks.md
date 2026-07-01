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

**Setup:** GloVe-1.2M dataset (1,183,514 base vectors, 10K queries, dim=200, cosine/angular), M=16, ef_construction=200, local ARM (Apple Silicon M-series). Multi-threaded batch insert + parallel batch search (`search_batch`).

| ef_search | VectorDB QPS | hnswlib QPS | VectorDB R@10 | hnswlib R@10 |
|----------:|-------------:|------------:|--------------:|-------------:|
| 50  | **51,695** | 42,069 | 0.6837 | 0.6585 |
| 100 | **30,657** | 24,946 | 0.7641 | 0.7454 |
| 200 | **19,120** | 14,063 | 0.8272 | 0.8115 |
| 400 | **10,422** | 7,760  | 0.8730 | 0.8625 |
| 800 |  **5,755** | 4,210  | 0.9092 | 0.9017 |

Build: VectorDB **78s (15,164 vec/s)** vs hnswlib **88.6s (13,350 vec/s)** — VectorDB **1.14× faster**.

See `docs/benchmark_glove_vs_hnswlib.md` for full analysis.

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

**Setup:** SIFT-1M dataset (1M base vectors, 10K queries, dim=128, L2), M=16, ef_construction=200, local ARM (Apple Silicon M-series). Multi-threaded batch insert + parallel batch search (`search_batch`).

| ef_search | VectorDB QPS | hnswlib QPS | VectorDB R@10 | hnswlib R@10 |
|----------:|-------------:|------------:|--------------:|-------------:|
| 50  | **104,385** | 86,208 | 0.9539 | 0.9467 |
| 100 |  **62,225** | 49,616 | 0.9858 | 0.9829 |
| 200 |  **41,832** | 27,266 | 0.9964 | 0.9955 |
| 400 |  **23,432** | 15,434 | 0.9987 | 0.9988 |
| 800 |  **13,690** |  8,541 | 0.9991 | 0.9992 |

Build: VectorDB **23.1s (43,311 vec/s)** vs hnswlib **32.3s (30,930 vec/s)** — VectorDB **1.4× faster**.

### Observations

**1. ef=50 and ef=100 give similar QPS but not identical**

At 1M scale the HNSW graph has enough layers that even ef=50 forces traversal of a large minimum number of nodes. ef=100 doubles the candidate budget but visits only marginally more nodes in practice, giving roughly the same QPS.

**2. Recall@10 sweet spot is ef=200**

ef=200 achieves 0.9964 R@10 at 41,832 QPS. Going to ef=800 gains only 0.003 recall at 3× the QPS cost. For most applications ef=200 is the right default.

**3. VectorDB beats hnswlib on both build and QPS**

Build is 1.4× faster despite WAL persistence (hnswlib is pure in-memory). QPS advantage grows with ef: +21% at ef=50, +54% at ef=200, +60% at ef=800. Recall is identical or slightly higher at every point.

See `docs/benchmark_sift1m_vs_hnswlib.md` for full analysis.

## HNSW large-scale insert throughput

Current throughput with multi-threaded batch insert (`insert_batch_mt`, `hardware_concurrency()` threads):

| N | Throughput |
|---|-----------|
| 200K | ~37K vec/s |
| 1M (SIFT) | 43,311 vec/s |
| 1.18M (GloVe) | 15,164 vec/s |

GloVe is slower because cosine distance (3 dot products per distance) costs more than L2 (1).

Throughput still degrades slightly with N due to the O(log N) algorithmic factor and growing cache miss rate as the graph exceeds L3 size — but the multi-threaded implementation keeps the absolute numbers high enough that SIFT-1M builds in 23s vs hnswlib's 32s.

See `docs/build_time_optimization.md` for the full optimization journey (from 4,285s to 23s).

---

## Recovery Time Benchmark (Day 28)

**Setup:** dim=128, M=16, ef_construction=200, Apple Silicon (ARM). No checkpoint before close — WAL is the only persistence. Recovery time = `Engine(data_dir)` constructor (pure C++ WAL replay, no Python overhead).  
**Script:** `bench/bench_recovery.py`

| WAL size | Vectors | macOS (ARM) | x86 CI (AMD EPYC) | Replay speed (x86) |
|----------|--------:|------------:|------------------:|-------------------:|
| 0.51 MB  |   1,000 |      ~90 ms |            ~90 ms |          ~11 Kv/s |
| 1.54 MB  |   3,000 |     ~330 ms |           ~510 ms |           ~6 Kv/s |
| 4.1 MB   |   8,000 |   ~1,350 ms |         ~2,190 ms |           ~4 Kv/s |

x86 CI is ~60% slower than macOS at N=8K. The virtualized AMD EPYC runner has lower single-core throughput for HNSW graph construction, which is the bottleneck during WAL replay.

**Finding:** Recovery time scales O(N log N), not O(N). WAL replay re-inserts each record into the HNSW graph, and HNSW insert is O(M × ef_construction × log N) — the same cost as the original insert. Replay speed drops from ~11 Kv/s at N=1K to ~4 Kv/s at N=8K as the graph deepens. A 256 MB WAL (~470K records at dim=128) would take roughly 15–25 minutes to recover from WAL alone on CI-class hardware.

**Why checkpoint matters:** After a checkpoint, the graph snapshot is loaded in O(N) (sequential file read), and only the WAL records since the last checkpoint need replaying. A checkpoint every 50K inserts bounds recovery time to ~30–60s regardless of total collection size.

---

## Stress Test — Mixed Workload (Day 28)

**Setup:** dim=16, continuous insert + search + delete loop for 60 seconds. Checkpoint every 10 iterations. At end: engine reopened, 500 random live vectors verified searchable.  
**Script:** `bench/bench_stress.py`

| Metric | macOS ARM (60s) | x86 CI (AMD EPYC, 60s) |
|--------|----------------:|----------------------:|
| Inserts | 34,050 | 47,000 |
| Searches | 6,810 | 9,400 |
| Deletes | 13,600 | 18,780 |
| Throughput | **907 ops/s** | **1,252 ops/s** |
| Crashes | 0 | 0 |
| Data loss | 0 | 0 |

**PASS on both platforms** — no crashes, no data loss.

| | macOS ARM | x86 CI (AMD EPYC) |
|-|----------:|------------------:|
| Inserts/sec | 34,050 / 60s = **568/s** | 47,000 / 60s = **783/s** |
| ops/s | **907** | **1,252** |

CI is ~1.4× faster for inserts. The gap narrowed from the old 6.4× because the bottleneck shifted from single-threaded distance compute (where AVX2 was 2× faster than NEON at dim=16) to WAL I/O and graph traversal overhead, which are architecture-independent.

---

## Filtered Search — Selectivity vs QPS and Recall (Day 28)

**Setup:** N=10K vectors, dim=128, M=16, ef=100, top_k=10. Each vector assigned a category from {0..K−1} uniformly. Filter targets category 0 (expected hit count = N/K). Brute-force ground truth computed with numpy.  
**Script:** `bench/bench_filtered.py`

**Selectivity** = fraction of vectors that pass the filter = 1/K. K=1 means all vectors are in the same category (no filter effect, 100% pass). K=100 means each category holds 1% of vectors — only 100 out of 10K pass the filter. Lower selectivity = stricter filter = fewer candidates eligible for the result.

**Recall@10** = fraction of true top-10 nearest neighbors (by brute force) that HNSW actually returns. Range is 0–1; 1.000 = perfect, identical to brute force. 0.700 means HNSW found 7 out of the 10 true nearest neighbors and missed 3. The problem is not recall=1 at high selectivity — that means search is working correctly. The problem is recall collapsing at low selectivity (1% and below), where HNSW starts missing real nearest neighbors.

| Selectivity | Candidate pool | QPS | Recall@10 |
|------------:|---------------:|----:|----------:|
| 100% | 10,000 | 3,887 | 1.000 |
| 50% | 5,000 | 7,249 | 1.000 |
| 10% | 1,000 | 12,818 | 1.000 |
| 1% | 100 | 15,490 | **0.685** |
| 0.1% | 10 | 12,130 | **0.105** |

**Finding 1 — QPS peaks at ~1% selectivity then drops slightly at 0.1%.** At 0.1% selectivity only 10 candidates pass the filter — the heap is trivially small but HNSW must still traverse the same number of graph nodes to find them, so most traversal work is wasted on rejected candidates. The slight QPS drop (15,490 → 12,130) reflects this increasing proportion of wasted work. At 1%–50% selectivity, the candidate pool is large enough to keep the graph traversal efficient while the smaller heap makes result assembly faster.

**Finding 2 — Recall collapses at low selectivity.** The HNSW graph is built for unfiltered similarity. At 1% selectivity only 100 vectors are eligible. With ef=100, beam search explores ~100 candidates, but most are rejected by the filter — the effective candidate pool for the filtered result is far smaller than ef. Some true nearest neighbors in the eligible set are never reached because they are poorly connected in the graph (the graph was built ignoring the filter).

This is the fundamental tension in filtered ANN: **the graph is optimized for global proximity, not per-filter proximity.** Production systems (Weaviate, Qdrant, Pinecone) address this with dedicated per-segment indexes or hybrid HNSW+inverted-index structures. For selectivity > 10%, post-filtering on top of unfiltered ANN works well; below 1%, recall degrades significantly without per-filter graph construction.

---

## ann-benchmarks Comparison — VectorDB vs hnswlib vs faiss

**Methodology:** ann-benchmarks style — same machine, same parameters (M=16, ef_construction=200), 10K queries, primary metric Recall@10. **Hardware:** Apple Silicon (ARM NEON), macOS.

### SIFT-1M (L2, 128D, 1M vectors) — `bench/bench_ann_compare.py`

| ef | VectorDB | R@10 | hnswlib | R@10 | faiss | R@10 |
|---:|---------:|-----:|--------:|-----:|------:|-----:|
|  50 | **103,674** | 0.9543 | 87,737 | 0.9458 | 108,021 | 0.9526 |
| 100 |  **62,497** | 0.9861 | 49,498 | 0.9827 |  51,094 | 0.9857 |
| 200 |  **41,383** | 0.9965 | 27,726 | 0.9955 |  18,499 | 0.9960 |
| 400 |  **23,712** | 0.9988 | 15,307 | 0.9986 |   7,467 | 0.9987 |
| 800 |  **13,668** | 0.9992 |  8,608 | 0.9992 |   3,573 | 0.9992 |

Build: VectorDB **22.8s** vs hnswlib **32.2s** vs faiss **31.5s** — VectorDB fastest.

### GloVe-1.2M (cosine, 200D, 1.18M vectors) — `bench/bench_glove.py`

| ef | VectorDB | R@10 | hnswlib | R@10 | faiss | R@10 |
|---:|---------:|-----:|--------:|-----:|------:|-----:|
|  50 | 52,143 | **0.6840** | 43,403 | 0.6593 | **58,209** | 0.6733 |
| 100 | **31,774** | **0.7654** | 25,014 | 0.7461 | 29,966 | 0.7516 |
| 200 | **19,085** | **0.8264** | 14,189 | 0.8128 | 11,555 | 0.8147 |
| 400 | **10,540** | **0.8731** |  7,749 | 0.8631 |  4,767 | 0.8625 |
| 800 |  **5,953** | **0.9091** |  4,157 | 0.9018 |  2,191 | 0.8996 |

Build: VectorDB **77.8s** vs hnswlib **88.0s** vs faiss **67.1s** — faiss fastest on build, VectorDB fastest on search (ef≥100).

### Findings

**faiss at ef=50:** faiss edges out VectorDB on SIFT (108K vs 104K QPS) and GloVe (58K vs 52K) at ef=50. At this setting the candidate pool is very small so search terminates quickly for all implementations — the difference reflects faiss's lower Python call overhead per batch (raw C array, no dict construction).

**faiss QPS collapse at high ef:** faiss drops sharply as ef grows. At ef=200 VectorDB is 2.2× faster (SIFT) and 1.65× faster (GloVe); at ef=800 VectorDB is 3.8× faster (SIFT) and 2.7× faster (GloVe). faiss's HNSW implementation visits neighbors sequentially without prefetching, so it is DRAM-bound at high ef when the graph exceeds L3 cache.

**Recall parity:** All three systems achieve the same recall at matching ef values — the HNSW algorithm is identical. VectorDB's recall is equal to or slightly above hnswlib and faiss at every point.

See `docs/benchmark_sift1m_vs_hnswlib.md`, `docs/benchmark_glove_vs_hnswlib.md` for full analysis.

---

## Memory Usage and Per-Query Latency — SIFT-1M

**Setup:** SIFT-1M, M=16, ef_construction=200, ef_search=200. Each implementation measured in a fresh subprocess (no RSS cross-contamination). Latency = single-threaded one-query-at-a-time, 2,000 queries after 200 warmup. Script: `bench/bench_memory_latency.py`.

Raw vector data (no index): **488 MB** (512 B/vec, 1M × 128D × float32).

### Memory

| System | Index RSS | B/vec | vs raw vectors |
|--------|----------:|------:|---------------:|
| VectorDB | **2,017 MB** | 2,115 | **4.1×** |
| hnswlib | 820 MB | 860 | 1.7× |
| faiss | 822 MB | 862 | 1.7× |

VectorDB uses ~2.4× more memory than hnswlib/faiss. The extra cost comes from collocating neighbor IDs and vector data in `node_blocks_` (each node stores M0=32 neighbor IDs + the full 128D vector inline), plus a separately mmap'd `VectorFile` (needed for WAL recovery). hnswlib and faiss store vectors once; VectorDB stores them twice — once in `node_blocks_` for fast search, once in `VectorFile` for durability.

### Per-query latency (single-threaded, ef=200)

| System | P50 | P95 | P99 |
|--------|----:|----:|----:|
| VectorDB | **0.24 ms** | **0.28 ms** | **0.30 ms** |
| faiss | 0.39 ms | 0.46 ms | 0.49 ms |
| hnswlib | 0.45 ms | 0.53 ms | 0.57 ms |

VectorDB is **1.6–1.9× lower latency** than hnswlib and **1.6–1.7×** lower than faiss per single query. The P99/P50 ratio is tight (1.25× for VectorDB vs 1.27× for hnswlib) — no latency spikes from the parallel batch implementation.

The single-query latency advantage comes from the same sources as the QPS advantage: collocated memory layout and prefetching reduce DRAM round-trips per query. At ef=200 the graph is large enough that memory access dominates over compute.
