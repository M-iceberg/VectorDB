// -----------------------------------------------------------------------------
// distance_neon.cpp — ARM NEON SIMD distance implementations  (Day 3)
//
// Compiled only on aarch64 (see src/core/CMakeLists.txt).
//
// Why NEON is faster than scalar (distance_naive.cpp):
//   Scalar general-purpose registers are 64-bit and hold one float at a time.
//   NEON vector registers are 128-bit and hold 4 floats simultaneously.
//   A single NEON instruction operates on all 4 lanes in parallel, so:
//     - Fewer load instructions: one vld1q_f32 issues 1 load instruction for
//       4 floats instead of 4 separate instructions. The actual memory traffic
//       is the same — both scalar and SIMD pull a full 64-byte cache line — but
//       SIMD uses 4x fewer instructions to do it.
//     - Fewer compute instructions: one vmlaq_f32 replaces 4 multiplies + 4 adds.
//     - Fewer loop iterations: the main loop steps by 4 instead of 1,
//       reducing loop-control overhead (increment, compare, branch) by 4x.
//   Fewer total instructions means the CPU pipeline stays fuller — less time
//   stalled waiting for the next instruction to decode and dispatch.
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
    // 4-accumulator unrolled NEON L2.
    //
    // Why 4 accumulators: a single vmlaq_f32 has ~3 cycle latency on Apple M-series.
    // With one accumulator every iteration depends on the previous result —
    // throughput is capped at 1 iteration per 3 cycles.  With 4 independent
    // accumulators the CPU's 4 NEON execution units can issue one FMA per cycle
    // to each lane, reaching ~4× the throughput for the same instruction count.
    // Each outer iteration handles 16 floats (4 groups × 4 lanes) so dim=128
    // completes in 8 outer iterations instead of 32.
    float compute(const float* a, const float* b, size_t dim) const override {
        float32x4_t s0 = vdupq_n_f32(0.0f);
        float32x4_t s1 = vdupq_n_f32(0.0f);
        float32x4_t s2 = vdupq_n_f32(0.0f);
        float32x4_t s3 = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 16 <= dim; i += 16) {
            float32x4_t d0 = vsubq_f32(vld1q_f32(a+i),    vld1q_f32(b+i));
            float32x4_t d1 = vsubq_f32(vld1q_f32(a+i+4),  vld1q_f32(b+i+4));
            float32x4_t d2 = vsubq_f32(vld1q_f32(a+i+8),  vld1q_f32(b+i+8));
            float32x4_t d3 = vsubq_f32(vld1q_f32(a+i+12), vld1q_f32(b+i+12));
            s0 = vmlaq_f32(s0, d0, d0);
            s1 = vmlaq_f32(s1, d1, d1);
            s2 = vmlaq_f32(s2, d2, d2);
            s3 = vmlaq_f32(s3, d3, d3);
        }
        // reduce 4 accumulators
        float32x4_t vsum = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
        // 4-float tail (covers dim % 16 remaining)
        for (; i + 4 <= dim; i += 4) {
            float32x4_t d = vsubq_f32(vld1q_f32(a+i), vld1q_f32(b+i));
            vsum = vmlaq_f32(vsum, d, d);
        }
        float result = vaddvq_f32(vsum);
        for (; i < dim; ++i) { float d = a[i]-b[i]; result += d*d; }
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
