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
