#include "distance.h"
#include <cmath>

namespace vectordb {

// Default batch: loop over compute(). SIMD subclasses override this.
void DistanceCompute::compute_batch(const float* query, const float* candidates,
                                    size_t n, size_t dim, float* out) const {
    for (size_t i = 0; i < n; ++i) {
        out[i] = compute(query, candidates + i * dim, dim);
    }
}

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

// Default factory — returns scalar implementation.
// Day 3: distance_neon.cpp will override this on ARM.
// Day 4: distance_dispatch.cpp will override this on x86.
DistanceCompute* DistanceCompute::create(Metric metric) {
    switch (metric) {
        case Metric::L2:           return new NaiveL2{};
        case Metric::Cosine:       return new NaiveCosine{};
        case Metric::InnerProduct: return new NaiveIP{};
    }
    return nullptr;
}

}  // namespace vectordb
