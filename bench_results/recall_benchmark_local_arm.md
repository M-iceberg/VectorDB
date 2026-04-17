# HNSW Parameter Sweep

**Dataset:** siftsmall (10K base vectors, 100 queries, dim=128, L2)
**Machine:** local ARM (Apple Silicon)
**k:** 10 (each search returns 10 nearest neighbors; recall@10 measures how many of those 10 match the brute-force ground truth)

| M | ef_construction | ef_search | recall@10 | QPS  |
|---|-----------------|-----------|-----------|------|
| 8  | 100 | 16  | 94.7% | 9930 |
| 8  | 100 | 32  | 98.3% | 6332 |
| 8  | 100 | 64  | 99.5% | 3796 |
| 8  | 100 | 128 | 100%  | 2228 |
| 16 | 200 | 16  | 97.0% | 7703 |
| 16 | 200 | 32  | 99.4% | 4809 |
| 16 | 200 | 64  | 99.7% | 2875 |
| 16 | 200 | 128 | 100%  | 1641 |
| 32 | 200 | 16  | 98.0% | 7057 |
| 32 | 200 | 32  | 99.3% | 4446 |
| 32 | 200 | 64  | 99.7% | 2685 |
| 32 | 200 | 128 | 100%  | 1606 |

## Takeaways

**ef_search is the main recall vs QPS knob.** Doubling ef_search roughly halves QPS and gains 1-3% recall. QPS measures how many search queries can be processed per second — higher is better.

**M controls graph density.** Higher M means more neighbors per node, which improves recall at low ef_search but increases build time and memory. M=32 shows little improvement over M=16 at this scale.

**Practical recommendations:**
- recall ≥ 99%, high throughput: M=16, ef_search=32 → 99.4% recall, 4809 QPS
- maximum recall: ef_search=128 → 100% recall, ~1600 QPS
- M=8 with ef_search=128 matches M=16 recall at higher QPS — smaller graphs traverse faster.

At 10K scale all configs reach 100% recall at ef_search=128. Differences will be more pronounced at 1M scale.
