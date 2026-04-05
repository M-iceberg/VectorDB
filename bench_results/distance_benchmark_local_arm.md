# Distance Compute Benchmark Summary

**Platform:** Apple M5 Pro (ARM aarch64)
**Build:** Release (-O3, -march=native)

AVX2 results will be added after x86 CI runs (trigger `bench.yml` manually).

## How to reproduce

```bash
cmake -B build_release -DVECTORDB_BENCH=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_release --target bench_distance
./build_release/bench/bench_distance --benchmark_out=bench_results/day5_full_local_arm.json --benchmark_out_format=json
```

---

## 1. Scalar vs NEON — single compute()

| Metric | dim | Scalar (ns) | NEON (ns) | Speedup |
|--------|-----|-------------|-----------|---------|
| L2 | 128 | 26.7 | 8.52 | 3.1x |
| L2 | 768 | 290 | 94.0 | 3.1x |
| L2 | 1536 | 656 | 237 | 2.8x |
| Cosine | 128 | 44.3 | 11.6 | 3.8x |
| Cosine | 768 | 363 | 104 | 3.5x |
| Cosine | 1536 | 763 | 258 | 3.0x |
| InnerProduct | 128 | 23.5 | 8.57 | 2.7x |
| InnerProduct | 768 | 279 | 91.7 | 3.0x |
| InnerProduct | 1536 | 642 | 228 | 2.8x |

Cosine shows the highest speedup because NEON parallelizes all three
accumulators (dot, na, nb) simultaneously in a single loop pass.

---

## 2. Scalar vs NEON — compute_batch (1024 candidates, dim=768)

| Metric | Scalar (ns) | NEON (ns) | Speedup |
|--------|-------------|-----------|---------|
| L2 | 297,028 | 96,267 | 3.1x |
| Cosine | 372,723 | 109,416 | 3.4x |
| InnerProduct | 286,506 | 93,873 | 3.1x |

---

## 3. Prefetch effect — compute_batch (1024 candidates, dim=768)

| Metric | No Prefetch (ns) | With Prefetch (ns) | Delta |
|--------|------------------|--------------------|-------|
| L2 | 96,657 | 96,213 | ~0% |
| Cosine | 107,571 | 107,098 | ~0% |
| InnerProduct | 94,073 | 93,961 | ~0% |

No measurable benefit on Apple Silicon — the hardware prefetcher already
detects sequential access patterns and prefetches automatically. Explicit
`__builtin_prefetch` is expected to help more on older x86 server CPUs.

---

## 4. Scalar tail — dims not divisible by 4 (L2)

NEON processes 4 floats per iteration; remaining elements fall back to scalar.

| dim | Scalar (ns) | NEON (ns) | Note |
|-----|-------------|-----------|------|
| 7 | 1.76 | 1.48 | 1 NEON iter + 3 scalar tail — minimal speedup |
| 769 | 291 | 94.8 | 192 NEON iters + 1 scalar tail — speedup preserved (3.1x) |

---

## 5. Unaligned memory — NEON with offset pointer

Pointer offset by 1 float (4 bytes) to break 16-byte alignment.
ARM vld1q_f32 supports unaligned loads with no penalty.

| Metric | dim | Aligned (ns) | Unaligned (ns) | Delta |
|--------|-----|--------------|----------------|-------|
| L2 | 128 | 8.52 | 8.77 | ~3% |
| L2 | 768 | 94.0 | 94.4 | ~0% |
| L2 | 1536 | 237 | 239 | ~1% |
| Cosine | 128 | 11.6 | 11.8 | ~2% |
| Cosine | 768 | 104 | 106 | ~2% |
| Cosine | 1536 | 258 | 262 | ~2% |
| InnerProduct | 128 | 8.57 | 8.66 | ~1% |
| InnerProduct | 768 | 91.7 | 92.0 | ~0% |
| InnerProduct | 1536 | 228 | 229 | ~0% |

---

## 6. x86 AVX2 results (TODO)

To be filled in after running `bench.yml` on the GitHub Actions x86 runner.
AVX2 processes 8 floats per iteration vs NEON's 4, so speedup over scalar
is expected to be higher (~4-6x for L2/IP, ~6-8x for Cosine).

---

## Raw JSON files

- `day3_local_arm.json` — Day 3 run (Scalar/NEON single + batch + scalar tail + unaligned)
- `day5_full_local_arm.json` — full run including prefetch comparison
