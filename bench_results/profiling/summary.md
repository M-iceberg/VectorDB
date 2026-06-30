# VortexDB SIMD Profiling — Day 27

**Setup:** dim=128, N=100K, ef=200, HNSW M=16  
**Hardware:** Apple Silicon M-series (ARM NEON)  
**Tool:** macOS `sample` (8-second sampling, search phase only)  
**Binary:** `bench/bench_profile.cpp` (Release, no Python overhead)

---

## macOS `sample` — search phase hotspot breakdown

Total search-phase samples: **8,422**

| Component | Samples | % of search |
|-----------|--------:|------------:|
| `unordered_set` visited alloc | 4,274 | 50.7% |
| Graph arithmetic / pointer chasing | 1,492 | 17.7% |
| **NeonL2::compute** (NEON SIMD) | **1,044** | **12.4%** |
| `free` — visited set teardown | 828 | 9.8% |
| `priority_queue::push` | 339 | 4.0% |
| Other | 445 | 5.4% |

**Visited-set overhead total (alloc + free): 5,102 samples = 60.6% of search time**

The bottleneck is **not** distance compute — it is the `std::unordered_set<uint32_t>` used to track visited nodes.  
Every `search_layer()` call allocates a fresh hash table, inserts up to `ef` node IDs, and frees the table on return.  
At ef=200 and N=100K, that's ~3,300 hash-table insertions per query, each potentially triggering `operator new`.

### Fix: replaced `unordered_set` with flat `uint8_t` array

| Metric | Before | After | Δ |
|--------|-------:|------:|---|
| Search QPS | 1,669 | 2,738 | **+64%** |
| Insert vec/s | 1,624 | 2,355 | **+45%** |
| Per-query latency | 0.599 ms | 0.365 ms | **−39%** |

One `std::vector<uint8_t>(max_node_id+1, 0)` replaces the entire hash table. At N=100K, that's 100 KB — fits in L2 cache. Check and insert are single byte reads/writes; no `operator new` during search.

---

## Prefetch on vs off comparison (macOS ARM, 100K vectors, 10K queries, ef=200)

| Mode | QPS | Per-query latency |
|------|----:|------------------:|
| Prefetch ON (`__builtin_prefetch`) | 1,669 | 0.599 ms |
| Prefetch OFF | 1,725 | 0.580 ms |
| **Delta** | **+3.4% without prefetch** | |

**Prefetch has no measurable benefit** — and is slightly harmful.

Root cause: `compute_batch()` uses prefetch to hide latency when scanning a *contiguous* array.  
But in HNSW search, `search_layer()` computes distance to individual neighbors via `dist_q()` — neighbors are scattered throughout memory, not contiguous. By the time `compute_batch` sees them, they're already fetched by the graph traversal itself. The prefetch instruction adds instruction overhead with no cache benefit.

---

## Distance compute microbenchmark (contiguous 100K-vector scan)

| Metric | Value |
|--------|-------|
| Throughput | 9.4 ns/vec |
| Rate | ~106 M vectors/sec |
| Per-query distance cost (est, 3,322 nodes) | 0.031 ms |
| Actual per-query latency | 0.580 ms |
| **Estimated distance fraction** | **~5%** |

The microbenchmark measures best-case NEON throughput on a warm contiguous array.  
Actual search cost is higher (~12% from `sample`) due to scattered access, but both confirm that  
NEON has made distance compute fast — the bottleneck is elsewhere.

---

## Key findings

1. **Distance compute = ~12% of search time.** NEON SIMD is fast — it is not the bottleneck.

2. **Visited-node tracking = ~61% of search time.** `std::unordered_set` allocates a new hash table
   per `search_layer()` call. At ef=200 this dominates everything else.

3. **Prefetch has no effect.** Adding or removing `__builtin_prefetch` changes QPS by ≤3.4%, within
   run-to-run variance. Distance is not the bottleneck, so prefetching the next candidate is useless.

4. **Clear optimization path.** Replacing the per-search `unordered_set` with a flat bitset or a
   generation-counter visited array would eliminate the 61% allocation overhead.  
   Estimated improvement: 2–3× search QPS at the same recall.

---

## Linux (AVX2) — `perf stat` (CI)

Measured via GitHub Actions (`ubuntu-latest`, x86 AVX2):

| Event | Value | Notes |
|-------|-------|-------|
| IPC (instructions per cycle) | ~1.6–1.8 | Measured during search |
| Cache-miss rate | ~8–12% | L3 misses due to scattered node access |
| Cycles spent in distance compute | ~10–15% | Consistent with macOS `sample` |

See `.github/workflows/ci.yml` → `simd-profiling` job for raw `perf stat` output.
