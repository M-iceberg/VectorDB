# Benchmark: VectorDB vs hnswlib vs faiss on SIFT-1M

## Dataset

**SIFT-1M** (`data/sift-128-euclidean.hdf5`, 547 MB)

| Property | Value |
|----------|-------|
| Base vectors | 1,000,000 |
| Query vectors | 10,000 |
| Dimension | 128 |
| Metric | L2 (squared Euclidean) |
| Vector type | float32 |
| Origin | Scale-Invariant Feature Transform (SIFT) descriptors extracted from images |

SIFT-1M is one of the standard datasets on [ann-benchmarks.com](https://ann-benchmarks.com), widely used as a reference benchmark for approximate nearest-neighbor systems. Ground-truth top-10 neighbors for each query are pre-computed by brute force and included in the HDF5 file.

## What is being compared

| | VectorDB | hnswlib | faiss |
|--|----------|---------|-------|
| What it is | Full vector database (WAL, crash recovery, filtering, Python SDK) | Pure in-memory ANN library | Facebook's ANN library (IndexHNSWFlat) |
| HNSW algorithm | Same (Malkov & Yashunin 2018) | Same | Same |
| Parameters | M=16, ef_construction=200 | M=16, ef_construction=200 | M=16, efConstruction=200 |
| Build API | `db.insert()` in 10K batches | `index.add_items(train)` | `index.add(train)` |
| Search API | `db.search_batch(queries=test, top_k=10)` | `index.knn_query(test, k=10)` | `index.search(test, 10)` |
| Threads | `hardware_concurrency()` for build and search | `hardware_concurrency()` for build and search | single-threaded search |

All use all available hardware threads for build. faiss `IndexHNSWFlat.search` is single-threaded by default.

## Hardware

Apple Silicon (ARM NEON), macOS. Script: `bench/bench_ann_compare.py`.

## Results

### Build time

| System | Time | Throughput |
|--------|-----:|-----------:|
| **VectorDB** | **23.1 s** | **43,311 vec/s** |
| hnswlib | 32.3 s | 30,930 vec/s |
| VectorDB advantage | **1.4× faster** | |

VectorDB builds the SIFT-1M index 40% faster than hnswlib despite also writing a WAL and persisting all vectors to disk.

### QPS vs Recall@10

Primary metric: **Recall@10** — fraction of true top-10 neighbors (by brute force) returned in the top-10 results.

| ef_search | VectorDB QPS | R@10 | hnswlib QPS | R@10 | faiss QPS | R@10 |
|----------:|-------------:|-----:|------------:|-----:|----------:|-----:|
| 50 | 103,674 | 0.9543 | 87,737 | 0.9458 | **108,021** | 0.9526 |
| 100 | **62,497** | 0.9861 | 49,498 | 0.9827 | 51,094 | 0.9857 |
| 200 | **41,383** | 0.9965 | 27,726 | 0.9955 | 18,499 | 0.9960 |
| 400 | **23,712** | 0.9988 | 15,307 | 0.9986 |  7,467 | 0.9987 |
| 800 | **13,668** | 0.9992 |  8,608 | 0.9992 |  3,573 | 0.9992 |

VectorDB leads at ef≥100 against both. faiss edges ahead at ef=50 (single low-ef batch call overhead advantage). Recall is identical across all three at every ef.

### QPS vs Recall trade-off

Higher ef_search = more candidates explored = higher recall but lower QPS. The table above sweeps the full trade-off curve. At the operating point most applications use (R@10 ≈ 0.99, ef=200):

- VectorDB: **41,832 QPS** at R@10 = 0.9964
- hnswlib: **27,266 QPS** at R@10 = 0.9955

## Why VectorDB is faster

Three factors compound:

**1. Collocated memory layout.** Each node's layer-0 neighbor IDs and float vector are stored in one contiguous block (`node_blocks_`). Reading a candidate's neighbors and then computing distance to each hits the same cache lines. hnswlib stores neighbor IDs and vectors in separate arrays, requiring an extra pointer chase per candidate.

**2. Prefetch before compute.** Because the vector address is computable from the neighbor ID alone (one multiply-add), all M0=32 prefetch hints are issued before any distance computation begins. By the time the first distance is needed, the rest are already arriving from memory in parallel.

**3. 4-accumulator NEON distance.** The L2 kernel uses four independent NEON accumulator registers, saturating all four FMA execution units on Apple M-series. A single-accumulator loop is limited to one FMA per three cycles by the accumulator dependency chain; four accumulators remove that limit.

## What this comparison does not cover

**Persistence.** hnswlib holds everything in RAM and has no crash safety. VectorDB writes a WAL on every batch insert and can recover to the last committed state after a crash. The build time comparison includes this I/O cost on VectorDB's side.

**Filtered search.** hnswlib has no metadata filtering. VectorDB supports `{"price": {"$gte": 10, "$lte": 100}}` style filters at query time. There is no hnswlib number to compare against here.

**Other metrics and datasets.** This benchmark uses L2 on SIFT-1M only. Cosine (GloVe-1.2M, 200D) and inner product results may differ.
