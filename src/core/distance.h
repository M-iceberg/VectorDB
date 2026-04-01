// -----------------------------------------------------------------------------
// distance.h — Abstract distance computation interface
//
// Defines the DistanceCompute base class and Metric enum used throughout the
// engine. All distance functions operate on float32 vectors.
//
// Implementations (selected by DistanceCompute::create()):
//   distance_naive.cpp   — scalar fallback, always compiled
//   distance_neon.cpp    — ARM NEON, compiled on aarch64 only
//   distance_avx2.cpp    — x86 AVX2 + runtime cpuid dispatch, compiled on x86_64 only
//
// API:
//   DistanceCompute::create(metric)
//       Factory. Returns the fastest available impl for this platform.
//       Caller owns the returned pointer.
//
//   compute(a, b, dim) → float
//       Distance between two vectors of length dim.
//       L2: squared Euclidean (no sqrt — cheaper for ranking).
//       Cosine: 1 − cosine_similarity  (0 = identical, 2 = opposite).
//       InnerProduct: −dot(a,b)         (lower = more similar).
//
//   compute_batch(query, candidates, n, dim, out)
//       Distances from one query to n contiguous candidates.
//       Default loops over compute(); SIMD subclasses override for throughput.
// -----------------------------------------------------------------------------
#pragma once
#include <cstddef>
#include <cstdint>

namespace vectordb {

enum class Metric : uint8_t {
    L2 = 0,
    Cosine = 1,
    InnerProduct = 2,
};

// Abstract distance compute interface.
// Implementations: distance_naive.cpp (scalar, always compiled),
//   distance_neon.cpp (ARM NEON, aarch64 only),
//   distance_avx2.cpp (x86 AVX2, x86_64 only).
// Use DistanceCompute::create() to get the best available impl.
class DistanceCompute {
public:
    virtual ~DistanceCompute() = default;

    // Returns the distance between two vectors of length `dim`.
    // L2 returns squared distance; Cosine returns 1 - cosine_sim; IP returns -dot.
    virtual float compute(const float* a, const float* b, size_t dim) const = 0;

    // Computes distances from `query` to each of `n` contiguous candidates.
    // Default implementation calls compute() in a loop; SIMD subclasses override.
    virtual void compute_batch(const float* query, const float* candidates,
                               size_t n, size_t dim, float* out) const;

    // Factory: returns the best available implementation for this platform.
    // On ARM, returns NEON-accelerated impl (distance_neon.cpp overrides this).
    // On x86, checks AVX2 support at runtime and returns AVX2 or scalar fallback.
    // Caller owns the returned pointer.
    static DistanceCompute* create(Metric metric);

    // Always returns the scalar (naive) implementation regardless of platform.
    // Used by SIMD unit tests to validate NEON/AVX2 results against the baseline.
    // Caller owns the returned pointer.
    static DistanceCompute* create_scalar(Metric metric);

    // Always returns the AVX2 implementation. Only callable on x86 builds;
    // used by unit tests to validate AVX2 results against the scalar baseline.
    // Caller owns the returned pointer.
    static DistanceCompute* create_avx2(Metric metric);
};

}  // namespace vectordb
