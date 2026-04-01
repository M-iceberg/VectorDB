// -----------------------------------------------------------------------------
// distance_naive.cpp — Scalar (portable) distance implementations
//
// Provides NaiveL2, NaiveCosine, and NaiveIP: plain C++ loops with no
// platform-specific intrinsics. Always compiled on all architectures and
// serves as the correctness baseline for SIMD implementations.
//
// Also defines the default DistanceCompute::create() factory, which returns
// these scalar implementations. On ARM, distance_neon.cpp overrides create()
// to return NEON-accelerated impls (Day 3). On x86, distance_dispatch.cpp
// overrides create() with a runtime cpuid check (Day 4).
//
// Distance semantics:
//   L2          — squared Euclidean: Σ(aᵢ−bᵢ)²   (no sqrt; cheaper for ranking)
//   Cosine      — 1 − dot(a,b) / (‖a‖·‖b‖)        (0=identical, 2=opposite)
//   InnerProduct— −dot(a,b)                         (lower = more similar)
// -----------------------------------------------------------------------------
#include "distance.h"
#include <cmath>

namespace vectordb {

// -----------------------------------------------------------------------------
// DistanceCompute — base class method implementations
// -----------------------------------------------------------------------------

// Default batch: loop over compute(). SIMD subclasses override this.
void DistanceCompute::compute_batch(const float* query, const float* candidates,
                                    size_t n, size_t dim, float* out) const {
    for (size_t i = 0; i < n; ++i) {
        out[i] = compute(query, candidates + i * dim, dim);
    }
}

// -----------------------------------------------------------------------------
// Scalar derived class implementations
// (must be defined before create() so new NaiveL2{} etc. can compile)
// -----------------------------------------------------------------------------

namespace {

class NaiveL2 final : public DistanceCompute {
public:
    float compute(const float* a, const float* b, size_t dim) const override {
        float sum = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            float d = a[i] - b[i];
            sum += d * d;
        }
        return sum;  // squared L2
    }
};

class NaiveCosine final : public DistanceCompute {
public:
    float compute(const float* a, const float* b, size_t dim) const override {
        float dot = 0.0f, na = 0.0f, nb = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            dot += a[i] * b[i];
            na  += a[i] * a[i];
            nb  += b[i] * b[i];
        }
        float denom = std::sqrt(na) * std::sqrt(nb);
        return denom > 1e-9f ? 1.0f - dot / denom : 1.0f;
    }
};

class NaiveIP final : public DistanceCompute {
public:
    float compute(const float* a, const float* b, size_t dim) const override {
        float dot = 0.0f;
        for (size_t i = 0; i < dim; ++i) dot += a[i] * b[i];
        return -dot;  // negated so lower = more similar
    }
};

}  // namespace

// create_scalar() — always returns the scalar implementation.
// Used by SIMD unit tests to get the baseline regardless of platform.
DistanceCompute* DistanceCompute::create_scalar(Metric metric) {
    switch (metric) {
        case Metric::L2:           return new NaiveL2{};
        case Metric::Cosine:       return new NaiveCosine{};
        case Metric::InnerProduct: return new NaiveIP{};
    }
    return nullptr;
}

// Default factory — returns scalar implementation on generic builds only.
// On ARM, distance_neon.cpp overrides this with a NEON-accelerated version.
// On x86, distance_dispatch.cpp overrides this with a runtime cpuid dispatch.
#if !defined(VECTORDB_ARCH_ARM) && !defined(VECTORDB_ARCH_X86)
DistanceCompute* DistanceCompute::create(Metric metric) {
    switch (metric) {
        case Metric::L2:           return new NaiveL2{};
        case Metric::Cosine:       return new NaiveCosine{};
        case Metric::InnerProduct: return new NaiveIP{};
    }
    return nullptr;
}
#endif  // !VECTORDB_ARCH_ARM && !VECTORDB_ARCH_X86

}  // namespace vectordb
