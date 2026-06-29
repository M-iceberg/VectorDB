# VortexDB Benchmark — GloVe-1.2M

**Dataset:** GloVe-1.2M (1,183,514 base vectors, 10,000 queries)  
**Dimension:** 200 · **Metric:** cosine (angular) · **Index:** HNSW M=16, ef_construction=200  
**Hardware:** Apple Silicon (ARM NEON) · **SDK:** Python (single-threaded)

## Insert throughput

| Metric | Value |
|--------|-------|
| Throughput | 146 vec/s |
| Total time | 135 min (8,112 s) |

## Search: recall vs QPS

| ef_search | QPS   | Recall@1 | Recall@10 | Recall@100 |
|----------:|------:|---------:|----------:|-----------:|
|        50 | 1,171 |   79.0%  |   76.8%   |   64.9%    |
|       100 | 1,172 |   79.0%  |   76.8%   |   64.9%    |
|       200 |   698 |   85.0%  |   82.6%   |   73.2%    |
|       400 |   408 |   89.6%  |   87.3%   |   79.7%    |
|       800 |   221 |   92.7%  |   90.9%   |   85.0%    |

**Sweet spot: ef=400** — 89.6% R@1 at 408 QPS (GloVe needs higher ef than SIFT to reach comparable recall).

## Key observations

- Recall is substantially lower than SIFT (85% vs 99% at ef=200): this is expected. GloVe-200-angular is one of the hardest standard ANN benchmarks. hnswlib achieves similar numbers (~85–95% R@1) on this dataset.
- Root cause: **hubness** (word embeddings in cosine space have many "hub" vectors that attract edges, leaving other regions poorly connected) + higher dimension (200 vs 128).
- ef=50 and ef=100 again plateau at identical recall and QPS — the minimum graph traversal cost is already saturated at ef=50.
- ef=800 needed to reach >90% R@1; on SIFT the same threshold is reached at ef=100.
- Insert throughput lower than SIFT (146 vs 177 vec/s): each distance computation is 56% larger (200 vs 128 floats).
