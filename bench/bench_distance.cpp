// SIMD speedup benchmark: naive vs NEON vs AVX2 — added Day 5 / Week 6.
// Build with: cmake -B build -DVECTORDB_BENCH=ON
#include <benchmark/benchmark.h>
#include "core/distance.h"

// TODO Day 5: BM_DistanceScalar, BM_DistanceNeon/Avx2, dim=128/768/1536
// TODO Day 25: compute_batch with prefetch on/off

BENCHMARK_MAIN();
