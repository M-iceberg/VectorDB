# Reproducible SIFT-1M ANN Comparison

The authoritative machine-readable output is
[`bench_results/ann_compare/results.json`](../bench_results/ann_compare/results.json).
It records every raw timing sample, the exact command, Git state, hardware,
thread count, compiler, and dependency versions.

## Methodology

- Dataset: SIFT-1M, 1,000,000 base vectors, 10,000 queries, 128D L2.
- Index settings: `M=16`, `efConstruction=200` for all systems.
- Concurrency: 18 threads explicitly passed to build and batch search.
- Search: 200 warm-up queries, then three full-query measurements per ef.
- Build: three clean index builds per system.
- Reported throughput: median; the JSON manifest also contains min/max.
- Compared APIs: Python-facing batch APIs, including each binding's result
  materialization overhead.

VectorDB currently fixes `M=16` and `efConstruction=200`. The harness rejects
non-default values instead of silently applying them only to competitors.

## Build results

| System | Samples (seconds) | Median | Throughput |
|---|---|---:|---:|
| **VectorDB** | 24.501, 24.497, 24.828 | **24.501 s** | **40,814 vec/s** |
| hnswlib | 32.672, 33.369, 31.959 | 32.672 s | 30,608 vec/s |
| Faiss | 31.761, 31.492, 31.648 | 31.648 s | 31,598 vec/s |

The VectorDB build includes WAL records, one synchronous group commit per 10K
batch, mmap vector publication, and parallel HNSW construction.

## Median QPS vs Recall@10

| efSearch | VectorDB QPS | R@10 | hnswlib QPS | R@10 | Faiss QPS | R@10 |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 101,287 | 0.9538 | 86,411 | 0.9467 | **110,015** | 0.9523 |
| 100 | **61,786** | 0.9859 | 49,444 | 0.9829 | 52,071 | 0.9854 |
| 200 | **40,209** | 0.9964 | 27,882 | 0.9955 | 18,686 | 0.9960 |
| 400 | **24,277** | 0.9987 | 15,364 | 0.9986 | 7,504 | 0.9987 |
| 800 | **13,546** | 0.9992 | 8,602 | 0.9993 | 3,559 | 0.9993 |

At ef=200, VectorDB measured 1.44× hnswlib QPS and 2.15× Faiss QPS at nearly
identical recall. This is one operating point, not a claim that VectorDB wins
every workload: Faiss leads at ef=50.

## Reproduction

```bash
python bench/bench_ann_compare.py \
  --threads 18 \
  --repeats 3 \
  --build-repeats 3 \
  --out bench_results/ann_compare
```

The benchmark atomically writes `results.json` and a QPS/recall plot. Generated
database files live below `bench_results/ann_compare/db/` and are ignored by
Git.

## Interpretation limits

- Same-ef comparison is useful here because recall is nearly matched, but a
  production evaluation should additionally interpolate QPS at fixed recall
  targets.
- Runs execute systems in a recorded fixed order; randomized-order runs can
  further reduce thermal/order bias.
- These are single-machine Apple Silicon results and should not be generalized
  to x86, other dimensions, or other metrics without rerunning the harness.
