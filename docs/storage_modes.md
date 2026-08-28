# Vector storage modes

VectorDB exposes two runtime layouts through `VECTORDB_STORAGE_MODE`:

- `performance` (default) colocates each vector with its fixed-capacity
  layer-0 adjacency block. `vectors.vdb` remains as the durable mmap copy.
- `compact` keeps only layer-0 adjacency in the HNSW block and computes
  distances directly against the contiguous mmap vector slots.

The on-disk format is compatible between modes. A database can be reopened in
either mode because graph snapshots retain vectors and `vectors.vdb` uses the
stable NodeId as its slot. A growing mmap can move its base address, so Engine
refreshes HNSW's external vector base after all stable-slot writes and before
graph construction or search becomes visible.

## Durability ordering

Both modes preserve the same commit protocol:

1. Reserve stable NodeIds.
2. Append all insert records to the WAL and `fdatasync` once.
3. Write vectors to their NodeId slots in `vectors.vdb`.
4. Construct the HNSW graph and publish metadata.

Checkpoint calls `msync` and `fsync` on populated vector slots before
publishing the new immutable graph generation. WAL truncation remains after
manifest publication, so a crash before vector persistence can still recover
from the retained WAL.

## SIFT-1M trade-off

Apple Silicon, 18 logical CPUs, `M=16`, `efConstruction=200`,
`efSearch=200`. Each latency/RSS row is the median of three clean-process
builds; each process executes 200 warm-up queries and 2,000 one-at-a-time
queries.

| System | Median RSS | P50 | P95 | P99 |
|---|---:|---:|---:|---:|
| VectorDB performance | 2,014 MB | 0.238 ms | 0.280 ms | 0.296 ms |
| VectorDB compact | 1,162 MB | 0.259 ms | 0.307 ms | 0.324 ms |
| hnswlib | 820 MB | 0.452 ms | 0.533 ms | 0.567 ms |
| Faiss | 823 MB | 0.387 ms | 0.456 ms | 0.479 ms |

Compact reduces VectorDB RSS by 42.3% and increases median P99 by 9.7% versus
performance mode. One compact run reached 0.529 ms P99; all samples are in
`bench_results/storage_tradeoff/results.json`.

The separate three-sample batch-search run measured compact mode at 37,962 QPS
and 0.9964 Recall@10 at `efSearch=200`, versus 27,801 QPS/0.9955 for hnswlib
and 18,456 QPS/0.9960 for Faiss. Its raw manifest is
`bench_results/ann_compare_compact/results.json`.

## Reproduction

```bash
python bench/bench_memory_latency.py \
  --threads 18 --repeats 3 \
  --vectordb-modes performance compact \
  --out bench_results/storage_tradeoff

python bench/bench_ann_compare.py \
  --threads 18 --repeats 3 --build-repeats 1 \
  --vectordb-storage-mode compact \
  --out bench_results/ann_compare_compact
```

Linux x86/AVX2 reproduction is defined in
`.github/workflows/x86-benchmark.yml`. It verifies AVX2 availability, pins the
SIFT-1M SHA-256, records `/proc` CPU flags and installed dependency versions,
and uploads raw JSON manifests and plots.

## Linux x86/AVX2 result

The full workflow completed on a 4-vCPU Intel Xeon Platinum 8370C Azure VM
with GCC 13.3 and a clean Git tree. At `efSearch=200`, performance mode reached
4,867 median QPS at 0.9966 Recall@10 and compact mode reached 4,683 QPS at
0.9967. In the performance-mode comparison, hnswlib reached 6,193 QPS at
0.9957 and Faiss reached 6,495 QPS at 0.9960.

This establishes correctness and reproducibility on AVX2, but not a
cross-platform performance win. VectorDB is 21% below hnswlib and 25% below
Faiss at this operating point. The committed manifests and runner information
are under `bench_results/linux_x86/`; the Apple/NEON and Linux/AVX2 results
must remain separately scoped in performance claims.
