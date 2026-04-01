# Benchmark Summary — GitHub Actions x86 / AVX2

**Source:** GitHub Actions CI (`bench.yml`), ubuntu-latest runner
**Date:** 2026-04-01
**Hardware:** 2 vCPU @ 3242 MHz, L1=32KB, L2=512KB, L3=32MB
**Build:** Release (-O3)
**Raw data:** `benchmark-x86.json`

---

## 1. Scalar vs AVX2 — single compute()

`BM_Simd` dispatches to AVX2 at runtime via `create()`; `BM_Avx2` calls
`create_avx2()` directly. Results are nearly identical, confirming the
runtime dispatch is working correctly.

| Metric | dim | Scalar (ns) | AVX2 (ns) | Speedup |
|--------|-----|-------------|-----------|---------|
| L2 | 128 | 75 | 12 | 6.2x |
| L2 | 768 | 673 | 86 | 7.8x |
| L2 | 1536 | 1392 | 206 | 6.8x |
| Cosine | 128 | 121 | 23 | 5.2x |
| Cosine | 768 | 718 | 112 | 6.4x |
| Cosine | 1536 | 1437 | 235 | 6.1x |
| InnerProduct | 128 | 72 | 14 | 5.0x |
| InnerProduct | 768 | 666 | 98 | 6.8x |
| InnerProduct | 1536 | 1388 | 221 | 6.3x |

AVX2 (8-wide) delivers roughly 2x the speedup of NEON (4-wide) as expected.

---

## 2. Scalar vs AVX2 — compute_batch (1024 candidates, dim=768)

`BM_SimdBatch` (dispatched) and `BM_Avx2Batch` (explicit) are identical.

| Metric | Scalar (ns) | AVX2 (ns) | Speedup |
|--------|-------------|-----------|---------|
| L2 | 689,241 | 93,608 | 7.4x |
| Cosine | 751,203 | 117,601 | 6.4x |
| InnerProduct | 685,729 | 102,791 | 6.7x |

---

## 3. Prefetch effect — compute_batch (1024 candidates, dim=768)

| Metric | No Prefetch (ns) | With Prefetch (ns) | Delta |
|--------|------------------|--------------------|-------|
| L2 | 92,967 | 93,608 | ~0% |
| Cosine | 118,232 | 117,481 | ~0% |
| InnerProduct | 102,777 | 102,474 | ~0% |

No measurable benefit. The x86 runner's hardware prefetcher handles sequential
access patterns automatically. `__builtin_prefetch` is expected to help on
older server CPUs with weaker hardware prefetchers.

---

## 4. Scalar tail — dims not divisible by 8 (L2)

AVX2 lane width = 8 floats.

| dim | Scalar (ns) | AVX2 (ns) | Note |
|-----|-------------|-----------|------|
| 7 | 3.44 | 3.44 | 0 AVX2 iters, all scalar tail (7 < 8) — no speedup |
| 769 | 670 | 89 | 96 AVX2 iters + 1 scalar tail — 7.5x speedup preserved |

dim=7 falls entirely into the scalar tail on x86 since 7 < 8.

---

## 5. Unaligned memory — AVX2 with offset pointer

Pointer offset by 1 float (4 bytes) to break natural alignment.

| Metric | dim | Aligned (ns) | Unaligned (ns) | Delta |
|--------|-----|--------------|----------------|-------|
| L2 | 128 | 12.1 | 12.2 | ~1% |
| L2 | 768 | 86.0 | 86.4 | ~0% |
| L2 | 1536 | 206.1 | 206.4 | ~0% |
| Cosine | 128 | 23.1 | 22.8 | ~0% |
| Cosine | 768 | 112.3 | 112.5 | ~0% |
| Cosine | 1536 | 234.6 | 233.0 | ~0% |
| InnerProduct | 128 | 14.4 | 14.3 | ~0% |
| InnerProduct | 768 | 97.6 | 98.9 | +1% |
| InnerProduct | 1536 | 220.7 | 219.8 | ~0% |

`_mm256_loadu_ps` handles unaligned loads — no measurable penalty on modern x86.
