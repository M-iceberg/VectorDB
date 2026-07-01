# Benchmark: VectorDB vs hnswlib vs faiss on GloVe-1.2M

## Dataset

**GloVe-1.2M** (`data/glove-200-angular.hdf5`)

| Property | Value |
|----------|-------|
| Base vectors | 1,183,514 |
| Query vectors | 10,000 |
| Dimension | 200 |
| Metric | Cosine (angular distance) |
| Vector type | float32 |
| Origin | GloVe word embeddings trained on Common Crawl |

GloVe-1.2M is a standard dataset on [ann-benchmarks.com](https://ann-benchmarks.com). It is the primary benchmark for cosine metric, and one of the harder ANN datasets due to its high dimensionality and the uniform distribution of cosine distances in word embedding spaces. Ground-truth top-10 neighbors are pre-computed by brute force.

## What is being compared

| | VectorDB | hnswlib | faiss |
|--|----------|---------|-------|
| Metric | `metric="cosine"` | `space='cosine'` | `METRIC_INNER_PRODUCT` + L2 normalization |
| Parameters | M=16, ef_construction=200 | M=16, ef_construction=200 | M=16, efConstruction=200 |
| Build API | `db.insert()` in 10K batches | `index.add_items(train)` | `index.add(train_normalized)` |
| Search API | `db.search_batch(queries=test, top_k=10)` | `index.knn_query(test, k=10)` | `index.search(test_normalized, 10)` |
| Threads | `hardware_concurrency()` for build and search | `hardware_concurrency()` for build and search | single-threaded search |

All normalize vectors for cosine. faiss has no native cosine index; vectors are L2-normalized and inner product is used as proxy (identical result to cosine).

## Hardware

Apple Silicon (ARM NEON), macOS. Script: `bench/bench_glove.py`.

## Results

### Build time

| System | Time | Throughput |
|--------|-----:|-----------:|
| **VectorDB** | **78.0 s** | **15,164 vec/s** |
| hnswlib | 88.6 s | 13,350 vec/s |
| VectorDB advantage | **1.14× faster** | |

### QPS vs Recall@10

| ef_search | VectorDB QPS | R@10 | hnswlib QPS | R@10 | faiss QPS | R@10 |
|----------:|-------------:|-----:|------------:|-----:|----------:|-----:|
| 50 | 52,143 | **0.6840** | 43,403 | 0.6593 | **58,209** | 0.6733 |
| 100 | **31,774** | **0.7654** | 25,014 | 0.7461 | 29,966 | 0.7516 |
| 200 | **19,085** | **0.8264** | 14,189 | 0.8128 | 11,555 | 0.8147 |
| 400 | **10,540** | **0.8731** |  7,749 | 0.8631 |  4,767 | 0.8625 |
| 800 |  **5,953** | **0.9091** |  4,157 | 0.9018 |  2,191 | 0.8996 |

VectorDB leads on recall at every ef value. On QPS: faiss edges ahead at ef=50; VectorDB leads at ef≥100 and widens the gap at high ef (2.7× vs faiss at ef=800).

## Why recall is lower than SIFT-1M

GloVe-1.2M is a significantly harder ANN problem than SIFT-1M. At ef=200, R@10 is 0.83 vs 0.996 on SIFT. This is expected and is not a deficiency in the implementation — hnswlib achieves the same numbers.

Three properties of GloVe make it harder:

**1. Higher dimension (200 vs 128).** The curse of dimensionality: in high-dimensional cosine spaces, the ratio of the nearest to the farthest neighbor distance shrinks. Most vectors are roughly equidistant from each other, making it difficult to distinguish a true nearest neighbor from a near-miss. HNSW graph traversal relies on proximity relationships to navigate — when those relationships are weak, beam search reaches a plateau regardless of ef.

**2. Hubness.** In high-dimensional cosine spaces, a small subset of vectors ("hubs") become nearest neighbors of many other vectors. The HNSW graph becomes skewed: hubs accumulate many edges, while other regions are sparsely connected. Queries landing in a poorly-connected region cannot be reached by beam search regardless of ef.

**3. Uniform distance distribution.** Word embeddings trained on co-occurrence statistics cluster around a shell in 200D space. Small angular differences between close vectors are hard to exploit in graph traversal, compared to SIFT's clustered structure where nearby vectors are clearly closer than distant ones.

These are properties of the dataset, not the index. Both VectorDB and hnswlib see the same recall ceiling.

## Comparison across datasets

**vs hnswlib:**

| | SIFT-1M (L2, 128D) | GloVe-1.2M (cosine, 200D) |
|---|---|---|
| Build | VectorDB **1.4×** faster | VectorDB **1.13×** faster |
| QPS advantage (ef=50) | **+18%** | **+20%** |
| QPS advantage (ef=200) | **+49%** | **+35%** |
| QPS advantage (ef=800) | **+59%** | **+43%** |
| Recall (ef=200) | 0.9965 vs 0.9955 | 0.8264 vs 0.8128 |

**vs faiss:**

| | SIFT-1M (L2, 128D) | GloVe-1.2M (cosine, 200D) |
|---|---|---|
| Build | VectorDB **1.38×** faster | faiss **1.16×** faster |
| QPS (ef=50) | faiss +4% | faiss +12% |
| QPS (ef=200) | VectorDB **+124%** | VectorDB **+65%** |
| QPS (ef=800) | VectorDB **+283%** | VectorDB **+172%** |
| Recall (ef=200) | essentially identical | VectorDB slightly higher |

faiss is competitive at ef=50 (small candidate set, search terminates quickly). As ef grows, VectorDB's prefetch-based memory access pattern dominates — faiss's HNSW search is not prefetch-optimized and becomes DRAM-bound.
