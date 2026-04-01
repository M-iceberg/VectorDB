# Day 3 — NEON vs Scalar Distance Benchmark

**Platform:** Apple Silicon (ARM aarch64)
**Build:** Release (-O3, -march=native)
**Date:** 2026-03-31

## How to reproduce

```bash
cmake -B build -DVECTORDB_BENCH=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target bench_distance
./build/bench/bench_distance --benchmark_out=bench_results/day3_neon.json --benchmark_out_format=json
```

Google Benchmark runs each case repeatedly until timing stabilizes, then
reports the median time per iteration. `SetItemsProcessed(iterations * dim)`
lets it report throughput in items/second alongside the latency.

## Single compute() — standard dims

| Metric | dim | Scalar (ns) | NEON (ns) | Speedup |
|--------|-----|-------------|-----------|---------|
| L2 | 128 | 29.4 | 10.6 | 2.8x |
| L2 | 768 | 310 | 109 | 2.8x |
| L2 | 1536 | 696 | 273 | 2.5x |
| Cosine | 128 | 58.2 | 13.5 | 4.3x |
| Cosine | 768 | 439 | 121 | 3.6x |
| Cosine | 1536 | 903 | 297 | 3.0x |
| InnerProduct | 128 | 27.4 | 10.3 | 2.7x |
| InnerProduct | 768 | 302 | 107 | 2.8x |
| InnerProduct | 1536 | 686 | 269 | 2.6x |

## compute_batch() — 1024 candidates, dim=768

| Metric | Scalar (ns) | NEON (ns) | Speedup |
|--------|-------------|-----------|---------|
| L2 | 332,512 | 114,076 | 2.9x |
| Cosine | 457,245 | 125,302 | 3.6x |
| InnerProduct | 318,439 | 110,117 | 2.9x |

## Scalar tail — dims not divisible by 4 (L2)

NEON processes 4 floats per iteration; remaining elements fall back to scalar.

| dim | Scalar (ns) | NEON (ns) | Note |
|-----|-------------|-----------|------|
| 7 | 1.97 | 2.11 | 1 NEON iter + 3 scalar tail — no speedup, slight overhead |
| 769 | 328 | 113 | 192 NEON iters + 1 scalar tail — speedup preserved (2.9x) |

dim=7 shows slight NEON overhead because nearly all work is in the scalar tail.
dim=769 is effectively the same as dim=768 — one extra scalar element is negligible.

## Unaligned memory — NEON with offset pointer (L2)

Pointer offset by 1 float (4 bytes) to break 16-byte alignment.
ARM vld1q_f32 supports unaligned loads; this confirms there is no meaningful penalty.

| dim | Aligned NEON (ns) | Unaligned NEON (ns) | Delta |
|-----|-------------------|---------------------|-------|
| 128 | 10.6 | 10.5 | ~0% |
| 768 | 109 | 109 | ~0% |
| 1536 | 273 | 290 | ~6% |

## Notes

- NEON processes 4 floats per instruction vs 1 for scalar.
- Cosine shows the highest speedup because it maintains 3 accumulators
  (dot, na, nb) — all three benefit from NEON parallelism simultaneously.
- Raw JSON results: `day3_neon.json`
