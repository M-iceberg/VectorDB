// -----------------------------------------------------------------------------
// distance_neon.cpp — ARM NEON SIMD distance implementations  (Day 3)
//
// Compiled only on aarch64 (see src/core/CMakeLists.txt).
//
// Why NEON is faster than scalar (distance_naive.cpp):
//   Scalar general-purpose registers are 64-bit and hold one float at a time.
//   NEON vector registers are 128-bit and hold 4 floats simultaneously.
//   A single NEON instruction operates on all 4 lanes in parallel, so:
//     - Fewer memory loads: one vld1q_f32 replaces 4 scalar loads.
//     - Fewer instructions: one vmlaq_f32 replaces 4 multiplies + 4 adds.
//     - Fewer loop iterations: the main loop steps by 4 instead of 1,
//       reducing loop-control overhead (increment, compare, branch) by 4x.
//   The accumulators (vsum, vdot, etc.) live in vector registers for the
//   entire loop, never touching memory until vaddvq_f32 collapses them
//   into a scalar at the end.
//
// Intrinsics used:
//   vld1q_f32  — load 4 floats from memory into a 128-bit register
//   vsubq_f32  — element-wise subtraction across 4 lanes
//   vmlaq_f32  — fused multiply-accumulate: acc = acc + a*b (4 lanes)
//   vaddvq_f32 — horizontal sum: collapse 4 lanes into one float
//
// Each function processes 4 floats per iteration, then falls back to a scalar
// tail loop for the remaining dim % 4 elements.
//
// Overrides DistanceCompute::create() on ARM, replacing the scalar factory in
// distance_naive.cpp (which guards itself with #ifndef VECTORDB_ARCH_ARM).
// -----------------------------------------------------------------------------
#include "distance.h"

#if defined(VECTORDB_ARCH_ARM)
#include <arm_neon.h>
#include <cmath>

namespace vectordb {

namespace {

// ----------------------------------------------------------------------------
// NeonL2 — squared Euclidean distance using NEON
// Returns Σ(aᵢ−bᵢ)² without sqrt (cheaper for ranking).
// ----------------------------------------------------------------------------
class NeonL2 final : public DistanceCompute {
public:
    // a and b each have `dim` elements. For each position i, compute (a[i]-b[i])^2
    // and sum them all up. NEON processes 4 positions per iteration instead of 1:
    //   Scalar: 8 iterations for dim=8, each handling a[i]-b[i] for one element.
    //   NEON:   2 iterations for dim=8, each handling [a[i..i+3]]-[b[i..i+3]] in parallel.
    float compute(const float* a, const float* b, size_t dim) const override {
        float32x4_t vsum = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 4 <= dim; i += 4) {
            float32x4_t va   = vld1q_f32(a + i);
            float32x4_t vb   = vld1q_f32(b + i);
            float32x4_t diff = vsubq_f32(va, vb);
            vsum = vmlaq_f32(vsum, diff, diff);  // vsum += diff * diff
        }
        float result = vaddvq_f32(vsum);  // horizontal sum of 4 lanes
        for (; i < dim; ++i) {            // scalar tail
            float d = a[i] - b[i];
            result += d * d;
        }
        return result;
    }
};

// ----------------------------------------------------------------------------
// NeonCosine — cosine distance = 1 − dot(a,b) / (‖a‖·‖b‖) using NEON
// Returns 0 for identical directions, 2 for opposite directions.
// Zero-vector guard: returns 1.0 when denominator ≈ 0.
// ----------------------------------------------------------------------------
class NeonCosine final : public DistanceCompute {
public:
    float compute(const float* a, const float* b, size_t dim) const override {
        float32x4_t vdot = vdupq_n_f32(0.0f);
        float32x4_t vna  = vdupq_n_f32(0.0f);
        float32x4_t vnb  = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 4 <= dim; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            vdot = vmlaq_f32(vdot, va, vb);  // vdot += a * b
            vna  = vmlaq_f32(vna,  va, va);  // vna  += a * a
            vnb  = vmlaq_f32(vnb,  vb, vb);  // vnb  += b * b
        }
        float dot    = vaddvq_f32(vdot);
        float na     = vaddvq_f32(vna);
        float nb     = vaddvq_f32(vnb);
        for (; i < dim; ++i) {              // scalar tail
            dot += a[i] * b[i];
            na  += a[i] * a[i];
            nb  += b[i] * b[i];
        }
        float denom = std::sqrt(na) * std::sqrt(nb);
        return denom > 1e-9f ? 1.0f - dot / denom : 1.0f;
    }
};

// ----------------------------------------------------------------------------
// NeonIP — inner product distance = −dot(a,b) using NEON
// Negated so that lower value = more similar (consistent with L2 / Cosine).
// ----------------------------------------------------------------------------
class NeonIP final : public DistanceCompute {
public:
    float compute(const float* a, const float* b, size_t dim) const override {
        float32x4_t vdot = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 4 <= dim; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            vdot = vmlaq_f32(vdot, va, vb);  // vdot += a * b
        }
        float dot = vaddvq_f32(vdot);
        for (; i < dim; ++i) dot += a[i] * b[i];  // scalar tail
        return -dot;
    }
};

}  // namespace

// ARM factory — overrides the guarded create() in distance_naive.cpp.
DistanceCompute* DistanceCompute::create(Metric metric) {
    switch (metric) {
        case Metric::L2:           return new NeonL2{};
        case Metric::Cosine:       return new NeonCosine{};
        case Metric::InnerProduct: return new NeonIP{};
    }
    return nullptr;
}

}  // namespace vectordb

#endif  // VECTORDB_ARCH_ARM
