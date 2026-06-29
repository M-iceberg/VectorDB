# VortexDB Benchmark — SIFT-1M

**Dataset:** SIFT-1M (1,000,000 base vectors, 10,000 queries)  
**Dimension:** 128 · **Metric:** L2 · **Index:** HNSW M=16, ef_construction=200  
**Hardware:** Apple Silicon (ARM NEON) · **SDK:** Python (single-threaded)

## Insert throughput

| Metric | Value |
|--------|-------|
| Throughput | 177 vec/s |
| Total time | 94 min (5,642 s) |

## Search: recall vs QPS

| ef_search | QPS   | Recall@1 | Recall@10 | Recall@100 |
|----------:|------:|---------:|----------:|-----------:|
|        50 | 1,909 |   98.6%  |   98.6%   |   93.3%    |
|       100 | 1,923 |   98.6%  |   98.6%   |   93.3%    |
|       200 | 1,148 |   99.2%  |   99.7%   |   98.2%    |
|       400 |   678 |   99.2%  |   99.9%   |   99.6%    |
|       800 |   390 |   99.3%  |   99.9%   |   99.9%    |

**Sweet spot: ef=200** — 99.2% R@1 at 1,148 QPS.

## Key observations

- ef=50 and ef=100 give identical recall and nearly identical QPS: the graph traversal at 1M scale has a minimum cost floor that ef=50 already hits.
- R@100 jumps from 93.3% (ef=100) to 98.2% (ef=200) because ef must be ≥ k to reliably find k neighbors.
- Recall@1 saturates at ~99.3% beyond ef=400; the remaining ~0.7% miss rate is the irreducible HNSW approximation error.
- QPS measured through Python SDK (pybind11 overhead per query); raw C++ throughput would be ~3–6× higher.
