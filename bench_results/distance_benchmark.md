# Distance Compute Benchmark Summary

**Platform:** Apple Silicon (ARM aarch64)
**Build:** Release (-O3, -march=native)

AVX2 results will be added after x86 CI runs (trigger `bench.yml` manually).

## How to reproduce

```bash
cmake -B build -DVECTORDB_BENCH=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target bench_distance
./build/bench/bench_distance --benchmark_out=bench_results/distance_raw.json --benchmark_out_format=json
```

---

## 1. Scalar vs NEON — single compute()

| Metric | dim | Scalar (ns) | NEON (ns) | Speedup |
|--------|-----|-------------|-----------|---------|
| L2 | 128 | 29.6 | 10.7 | 2.8x |
| L2 | 768 | 311 | 111 | 2.8x |
| L2 | 1536 | 695 | 279 | 2.5x |
| Cosine | 128 | 58.7 | 13.5 | 4.3x |
| Cosine | 768 | 449 | 121 | 3.7x |
| Cosine | 1536 | 923 | 302 | 3.1x |
| InnerProduct | 128 | 27.7 | 17.5 | 1.6x |
| InnerProduct | 768 | 306 | 115 | 2.7x |
| InnerProduct | 1536 | 694 | 261 | 2.7x |

Cosine shows the highest speedup because NEON parallelizes all three
accumulators (dot, na, nb) simultaneously in a single loop pass.

---

## 2. Scalar vs NEON — compute_batch (1024 candidates, dim=768)

| Metric | Scalar (ns) | NEON (ns) | Speedup |
|--------|-------------|-----------|---------|
| L2 | 332,466 | 113,579 | 2.9x |
| Cosine | 460,643 | 125,628 | 3.7x |
| InnerProduct | 331,197 | 117,804 | 2.8x |

---

## 3. Prefetch effect — compute_batch (1024 candidates, dim=768)

| Metric | No Prefetch (ns) | With Prefetch (ns) | Delta |
|--------|------------------|--------------------|-------|
| L2 | 113,999 | 113,604 | ~0% |
| Cosine | 125,364 | 125,437 | ~0% |
| InnerProduct | 118,041 | 117,834 | ~0% |

No measurable benefit on Apple Silicon — the hardware prefetcher already
detects sequential access patterns and prefetches automatically. Explicit
`__builtin_prefetch` is expected to help more on older x86 server CPUs.

---

## 4. Scalar tail — dims not divisible by 4 (L2)

NEON processes 4 floats per iteration; remaining elements fall back to scalar.

| dim | Scalar (ns) | NEON (ns) | Note |
|-----|-------------|-----------|------|
| 7 | 1.77 | 1.81 | 1 NEON iter + 3 scalar tail — no speedup, slight overhead |
| 769 | 329 | 111 | 192 NEON iters + 1 scalar tail — speedup preserved (3.0x) |

---

## 5. Unaligned memory — NEON with offset pointer

Pointer offset by 1 float (4 bytes) to break 16-byte alignment.
ARM vld1q_f32 supports unaligned loads with no penalty.

| Metric | dim | Aligned (ns) | Unaligned (ns) | Delta |
|--------|-----|--------------|----------------|-------|
| L2 | 128 | 10.7 | 10.5 | ~0% |
| L2 | 768 | 111 | 111 | ~0% |
| L2 | 1536 | 279 | 281 | ~1% |
| Cosine | 128 | 13.5 | 13.4 | ~0% |
| Cosine | 768 | 121 | 123 | ~2% |
| Cosine | 1536 | 302 | 300 | ~0% |

---

## 6. x86 AVX2 results (TODO)

To be filled in after running `bench.yml` on the GitHub Actions x86 runner.
AVX2 processes 8 floats per iteration vs NEON's 4, so speedup over scalar
is expected to be higher (~4-6x for L2/IP, ~6-8x for Cosine).

---

## Raw JSON files

- `day3_neon.json` — original Day 3 run
- `day5_full.json` — full run including prefetch comparison
