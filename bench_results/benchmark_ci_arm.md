# Benchmark Summary — GitHub Actions ARM / NEON

**Source:** GitHub Actions CI (`bench.yml`), ubuntu-24.04-arm runner
**Date:** 2026-04-01
**Hardware:** 2 vCPU @ 2000 MHz, L1=64KB, L2=1MB, L3=128MB
**Build:** Release (-O3)
**Raw data:** `benchmark_ci_arm.json`

---

## 1. Scalar vs NEON — single compute()

| Metric | dim | Scalar (ns) | NEON (ns) | Speedup |
|--------|-----|-------------|-----------|---------|
| L2 | 128 | 59 | 19 | 3.1x |
| L2 | 768 | 388 | 116 | 3.3x |
| L2 | 1536 | 784 | 233 | 3.4x |
| Cosine | 128 | 170 | 41 | 4.1x |
| Cosine | 768 | 1019 | 214 | 4.8x |
| Cosine | 1536 | 2039 | 420 | 4.9x |
| InnerProduct | 128 | 58 | 18 | 3.2x |
| InnerProduct | 768 | 385 | 113 | 3.4x |
| InnerProduct | 1536 | 781 | 228 | 3.4x |

Cosine shows the highest speedup because NEON parallelizes all three
accumulators (dot, na, nb) in a single loop pass.

---

## 2. Scalar vs NEON — compute_batch (1024 candidates, dim=768)

| Metric | Scalar (ns) | NEON (ns) | Speedup |
|--------|-------------|-----------|---------|
| L2 | 397,890 | 120,227 | 3.3x |
| Cosine | 1,044,538 | 219,280 | 4.8x |
| InnerProduct | 395,063 | 117,467 | 3.4x |

---

## 3. Prefetch effect — compute_batch (1024 candidates, dim=768)

| Metric | No Prefetch (ns) | With Prefetch (ns) | Delta |
|--------|------------------|--------------------|-------|
| L2 | 120,215 | 120,227 | ~0% |
| Cosine | 219,227 | 219,280 | ~0% |
| InnerProduct | 117,344 | 117,467 | ~0% |

No measurable benefit. The ARM runner's hardware prefetcher handles sequential
access patterns automatically.

---

## 4. Scalar tail — dims not divisible by 4 (L2)

NEON lane width = 4 floats.

| dim | Scalar (ns) | NEON (ns) | Note |
|-----|-------------|-----------|------|
| 7 | 3.54 | 2.95 | 1 NEON iter + 3 scalar tail — minimal speedup |
| 769 | 390 | 118 | 192 NEON iters + 1 scalar tail — 3.3x speedup preserved |

---

## 5. Unaligned memory — NEON with offset pointer

Pointer offset by 1 float (4 bytes) to break 16-byte alignment.

| Metric | dim | Aligned (ns) | Unaligned (ns) | Delta |
|--------|-----|--------------|----------------|-------|
| L2 | 128 | 19.3 | 19.7 | +2% |
| L2 | 768 | 116.4 | 117.2 | ~1% |
| L2 | 1536 | 233.2 | 234.6 | ~1% |
| Cosine | 128 | 41.1 | 40.8 | ~0% |
| Cosine | 768 | 214.1 | 211.5 | ~0% |
| Cosine | 1536 | 419.6 | 415.4 | ~0% |
| InnerProduct | 128 | 18.3 | 18.3 | ~0% |
| InnerProduct | 768 | 112.9 | 115.1 | +2% |
| InnerProduct | 1536 | 228.4 | 236.1 | +3% |

ARM `vld1q_f32` supports unaligned loads natively — negligible penalty.
