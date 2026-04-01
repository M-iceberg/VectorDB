// -----------------------------------------------------------------------------
// distance_avx2.cpp — x86 AVX2 SIMD distance implementations 
//
// Compiled only on x86_64 (see src/core/CMakeLists.txt).
//
// Why AVX2 is faster than scalar (distance_naive.cpp):
//   Scalar general-purpose registers are 64-bit and hold one float at a time.
//   AVX2 vector registers (ymm0-ymm15) are 256-bit and hold 8 floats simultaneously.
//   A single AVX2 instruction operates on all 8 lanes in parallel, so:
//     - Fewer load instructions: one _mm256_loadu_ps issues 1 load instruction
//       for 8 floats instead of 8 separate instructions. The actual memory traffic
//       is the same — both scalar and SIMD pull a full 64-byte cache line — but
//       SIMD uses 8x fewer instructions to do it.
//     - Fewer compute instructions: one _mm256_fmadd_ps replaces 8 multiplies + 8 adds.
//     - Fewer loop iterations: the main loop steps by 8 instead of 1,
//       reducing loop-control overhead (increment, compare, branch) by 8x.
//   Fewer total instructions means the CPU pipeline stays fuller — less time
//   stalled waiting for the next instruction to decode and dispatch.
//   The accumulators live in ymm registers for the entire loop, never touching
//   memory until hsum256() collapses all 8 lanes into a scalar at the end.
//
//   Concretely for L2 with dim=8:
//     Scalar: 8 iterations, each handling a[i]-b[i] for one element.
//     AVX2:   1 iteration,  handling [a[0..7]]-[b[0..7]] in parallel.
//
// Intrinsics used:
//   _mm256_loadu_ps   — load 8 floats from memory (u = unaligned, no penalty on modern x86)
//   _mm256_sub_ps     — element-wise subtraction across 8 lanes
//   _mm256_fmadd_ps   — fused multiply-add: acc = a*b + acc (8 lanes, requires -mfma)
//   _mm256_add_ps     — element-wise addition across 8 lanes
//   hsum256()         — horizontal sum: collapse 8 lanes into one float (see below)
//
// Each function processes 8 floats per iteration, then falls back to a scalar
// tail loop for the remaining dim % 8 elements.
//
// create_avx2() is defined here and declared in distance.h so that
// distance_dispatch.cpp and unit tests can access AVX2 impls directly.
// DistanceCompute::create() is defined in distance_dispatch.cpp, which
// selects AVX2 or scalar at runtime via __builtin_cpu_supports("avx2").
// -----------------------------------------------------------------------------
#include "distance.h"

#if defined(VECTORDB_ARCH_X86)
#include <immintrin.h>
#include <cmath>

namespace vectordb {

namespace {

// Horizontal sum: collapse all 8 lanes of a __m256 into one float.
// AVX2 has no single instruction for this; we extract two 128-bit halves,
// add them together, then use two SSE _mm_hadd_ps passes to finish.
inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);     // lower 4 lanes
    __m128 hi = _mm256_extractf128_ps(v, 1);  // upper 4 lanes
    __m128 s  = _mm_add_ps(lo, hi);            // pairwise add → 4 lanes
    s = _mm_hadd_ps(s, s);                     // horizontal add → 2 lanes
    s = _mm_hadd_ps(s, s);                     // horizontal add → 1 lane
    return _mm_cvtss_f32(s);
}

// ----------------------------------------------------------------------------
// Avx2L2 — squared Euclidean distance using AVX2
// Returns Σ(aᵢ−bᵢ)² without sqrt (cheaper for ranking).
// ----------------------------------------------------------------------------
class Avx2L2 final : public DistanceCompute {
public:
    // a and b each have `dim` elements. For each position i, compute (a[i]-b[i])^2
    // and sum them all up. AVX2 processes 8 positions per iteration instead of 1:
    //   Scalar: 8 iterations for dim=8, each handling a[i]-b[i] for one element.
    //   AVX2:   1 iteration  for dim=8, handling [a[0..7]]-[b[0..7]] in parallel.
    float compute(const float* a, const float* b, size_t dim) const override {
        __m256 vsum = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= dim; i += 8) {
            __m256 va   = _mm256_loadu_ps(a + i);
            __m256 vb   = _mm256_loadu_ps(b + i);
            __m256 diff = _mm256_sub_ps(va, vb);
            vsum = _mm256_fmadd_ps(diff, diff, vsum);  // vsum += diff * diff
        }
        float result = hsum256(vsum);
        for (; i < dim; ++i) {  // scalar tail
            float d = a[i] - b[i];
            result += d * d;
        }
        return result;
    }
};

// ----------------------------------------------------------------------------
// Avx2Cosine — cosine distance = 1 − dot(a,b) / (‖a‖·‖b‖) using AVX2
// Returns 0 for identical directions, 2 for opposite directions.
// Zero-vector guard: returns 1.0 when denominator ≈ 0.
// ----------------------------------------------------------------------------
class Avx2Cosine final : public DistanceCompute {
public:
    float compute(const float* a, const float* b, size_t dim) const override {
        __m256 vdot = _mm256_setzero_ps();
        __m256 vna  = _mm256_setzero_ps();
        __m256 vnb  = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= dim; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            vdot = _mm256_fmadd_ps(va, vb, vdot);  // vdot += a * b
            vna  = _mm256_fmadd_ps(va, va, vna);   // vna  += a * a
            vnb  = _mm256_fmadd_ps(vb, vb, vnb);   // vnb  += b * b
        }
        float dot = hsum256(vdot);
        float na  = hsum256(vna);
        float nb  = hsum256(vnb);
        for (; i < dim; ++i) {  // scalar tail
            dot += a[i] * b[i];
            na  += a[i] * a[i];
            nb  += b[i] * b[i];
        }
        float denom = std::sqrt(na) * std::sqrt(nb);
        return denom > 1e-9f ? 1.0f - dot / denom : 1.0f;
    }
};

// ----------------------------------------------------------------------------
// Avx2IP — inner product distance = −dot(a,b) using AVX2
// Negated so that lower value = more similar (consistent with L2 / Cosine).
// ----------------------------------------------------------------------------
class Avx2IP final : public DistanceCompute {
public:
    float compute(const float* a, const float* b, size_t dim) const override {
        __m256 vdot = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= dim; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            vdot = _mm256_fmadd_ps(va, vb, vdot);  // vdot += a * b
        }
        float dot = hsum256(vdot);
        for (; i < dim; ++i) dot += a[i] * b[i];  // scalar tail
        return -dot;
    }
};

}  // namespace

// create_avx2() — always returns the AVX2 implementation.
// Used by unit tests to validate AVX2 results against the scalar baseline.
DistanceCompute* DistanceCompute::create_avx2(Metric metric) {
    switch (metric) {
        case Metric::L2:           return new Avx2L2{};
        case Metric::Cosine:       return new Avx2Cosine{};
        case Metric::InnerProduct: return new Avx2IP{};
    }
    return nullptr;
}

// x86 factory — checks AVX2 support at runtime and returns the appropriate impl.
// A binary compiled with -mavx2 would crash on older CPUs without AVX2;
// __builtin_cpu_supports("avx2") detects the actual CPU at runtime so a single
// binary works on both old and new hardware.
DistanceCompute* DistanceCompute::create(Metric metric) {
    if (__builtin_cpu_supports("avx2")) {
        return create_avx2(metric);
    }
    return create_scalar(metric);
}

}  // namespace vectordb

#endif  // VECTORDB_ARCH_X86
