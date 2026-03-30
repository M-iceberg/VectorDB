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
    // Caller owns the returned pointer.
    static DistanceCompute* create(Metric metric);
};

}  // namespace vectordb
