#include <gtest/gtest.h>
#include "core/distance.h"
#include <cmath>
#include <memory>
#include <random>
#include <vector>

// Day 2: correctness baseline for the scalar (naive) distance implementations.
// Days 3–4: NEON / AVX2 impls will be validated against these same cases.

namespace vectordb {
namespace {

// --------------------------------------------------------------------------
// Double-precision reference — ground truth for random-vector tests
// --------------------------------------------------------------------------

double ref_l2(const float* a, const float* b, size_t dim) {
    double s = 0.0;
    for (size_t i = 0; i < dim; ++i) { double d = a[i] - b[i]; s += d * d; }
    return s;
}

double ref_cosine(const float* a, const float* b, size_t dim) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    double denom = std::sqrt(na) * std::sqrt(nb);
    return denom > 1e-9 ? 1.0 - dot / denom : 1.0;
}

double ref_ip(const float* a, const float* b, size_t dim) {
    double dot = 0.0;
    for (size_t i = 0; i < dim; ++i) dot += (double)a[i] * b[i];
    return -dot;
}

std::vector<float> random_vec(size_t dim, uint64_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

// --------------------------------------------------------------------------
// L2 — hand-crafted cases
// --------------------------------------------------------------------------

TEST(NaiveL2, ZeroDistance) {
    float v[] = {1.0f, 2.0f, 3.0f};
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::L2));
    EXPECT_FLOAT_EQ(dc->compute(v, v, 3), 0.0f);
}

TEST(NaiveL2, OrthogonalUnitVectors) {
    float a[] = {1.0f, 0.0f};
    float b[] = {0.0f, 1.0f};
    // (1-0)^2 + (0-1)^2 = 2
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::L2));
    EXPECT_FLOAT_EQ(dc->compute(a, b, 2), 2.0f);
}

TEST(NaiveL2, SingleDim) {
    float a[] = {3.0f};
    float b[] = {7.0f};
    // (3-7)^2 = 16
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::L2));
    EXPECT_FLOAT_EQ(dc->compute(a, b, 1), 16.0f);
}

TEST(NaiveL2, KnownThreeDim) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    // (1-4)^2 + (2-5)^2 + (3-6)^2 = 9+9+9 = 27
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::L2));
    EXPECT_FLOAT_EQ(dc->compute(a, b, 3), 27.0f);
}

// --------------------------------------------------------------------------
// Cosine — hand-crafted cases
// --------------------------------------------------------------------------

TEST(NaiveCosine, IdenticalVectors) {
    float a[] = {1.0f, 2.0f, 3.0f};
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::Cosine));
    EXPECT_NEAR(dc->compute(a, a, 3), 0.0f, 1e-6f);
}

TEST(NaiveCosine, Orthogonal) {
    float a[] = {1.0f, 0.0f};
    float b[] = {0.0f, 1.0f};
    // cosine_sim = 0 → distance = 1
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::Cosine));
    EXPECT_NEAR(dc->compute(a, b, 2), 1.0f, 1e-6f);
}

TEST(NaiveCosine, AntiParallel) {
    float a[] = {1.0f, 0.0f};
    float b[] = {-1.0f, 0.0f};
    // cosine_sim = -1 → distance = 2
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::Cosine));
    EXPECT_NEAR(dc->compute(a, b, 2), 2.0f, 1e-6f);
}

TEST(NaiveCosine, ScaleInvariant) {
    float a[] = {1.0f, 1.0f};
    float b[] = {2.0f, 2.0f};  // same direction, different magnitude
    float c[] = {3.0f, 3.0f};
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::Cosine));
    EXPECT_NEAR(dc->compute(a, b, 2), 0.0f, 1e-6f);
    EXPECT_NEAR(dc->compute(a, c, 2), 0.0f, 1e-6f);
}

TEST(NaiveCosine, Known45Degrees) {
    // [1,0] vs [1,1]/sqrt(2): cosine_sim = 1/sqrt(2) ≈ 0.7071
    float a[] = {1.0f, 0.0f};
    float b[] = {1.0f, 1.0f};
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::Cosine));
    float expected = 1.0f - 1.0f / std::sqrt(2.0f);
    EXPECT_NEAR(dc->compute(a, b, 2), expected, 1e-6f);
}

// --------------------------------------------------------------------------
// Inner Product — hand-crafted cases
// --------------------------------------------------------------------------

TEST(NaiveIP, KnownDot) {
    float a[] = {1.0f, 2.0f};
    float b[] = {3.0f, 4.0f};
    // dot = 1*3 + 2*4 = 11 → returns -11
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::InnerProduct));
    EXPECT_FLOAT_EQ(dc->compute(a, b, 2), -11.0f);
}

TEST(NaiveIP, Orthogonal) {
    float a[] = {1.0f, 0.0f};
    float b[] = {0.0f, 1.0f};
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::InnerProduct));
    EXPECT_FLOAT_EQ(dc->compute(a, b, 2), 0.0f);
}

TEST(NaiveIP, SelfDot) {
    float a[] = {3.0f, 4.0f};
    // dot = 9+16 = 25 → returns -25
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::InnerProduct));
    EXPECT_FLOAT_EQ(dc->compute(a, a, 2), -25.0f);
}

// --------------------------------------------------------------------------
// Random vector tests: float impl vs double-precision reference
// --------------------------------------------------------------------------

class NaiveDistanceRandom : public ::testing::TestWithParam<size_t> {};

TEST_P(NaiveDistanceRandom, L2VsReference) {
    size_t dim = GetParam();
    auto a = random_vec(dim, 42);
    auto b = random_vec(dim, 99);
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::L2));
    float got = dc->compute(a.data(), b.data(), dim);
    float ref = static_cast<float>(ref_l2(a.data(), b.data(), dim));
    EXPECT_NEAR(got, ref, ref * 1e-5f + 1e-6f);
}

TEST_P(NaiveDistanceRandom, CosineVsReference) {
    size_t dim = GetParam();
    auto a = random_vec(dim, 42);
    auto b = random_vec(dim, 99);
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::Cosine));
    float got = dc->compute(a.data(), b.data(), dim);
    float ref = static_cast<float>(ref_cosine(a.data(), b.data(), dim));
    EXPECT_NEAR(got, ref, 1e-5f);
}

TEST_P(NaiveDistanceRandom, IPVsReference) {
    size_t dim = GetParam();
    auto a = random_vec(dim, 42);
    auto b = random_vec(dim, 99);
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::InnerProduct));
    float got = dc->compute(a.data(), b.data(), dim);
    float ref = static_cast<float>(ref_ip(a.data(), b.data(), dim));
    EXPECT_NEAR(got, ref, std::abs(ref) * 1e-5f + 1e-6f);
}

// Dims: 1 (boundary), 3 (non-multiple of 4), 4, 7, 16, 128, 768, 1536
// Non-multiples of 4 are important for Day 3 NEON tail handling.
INSTANTIATE_TEST_SUITE_P(Dims, NaiveDistanceRandom,
    ::testing::Values(1, 3, 4, 7, 16, 128, 768, 1536));

// --------------------------------------------------------------------------
// compute_batch consistency: must match individual compute() calls
// --------------------------------------------------------------------------

TEST(NaiveBatch, MatchesSingleCalls) {
    constexpr size_t dim = 64;
    constexpr size_t n   = 8;

    auto query = random_vec(dim, 1);
    std::vector<float> candidates(dim * n);
    for (size_t i = 0; i < n; ++i) {
        auto v = random_vec(dim, 100 + i);
        std::copy(v.begin(), v.end(), candidates.data() + i * dim);
    }

    for (Metric metric : {Metric::L2, Metric::Cosine, Metric::InnerProduct}) {
        auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(metric));
        std::vector<float> batch_out(n);
        dc->compute_batch(query.data(), candidates.data(), n, dim, batch_out.data());

        for (size_t i = 0; i < n; ++i) {
            float single = dc->compute(query.data(), candidates.data() + i * dim, dim);
            EXPECT_FLOAT_EQ(batch_out[i], single)
                << "metric=" << static_cast<int>(metric) << " i=" << i;
        }
    }
}

// --------------------------------------------------------------------------
// compute_batch edge cases
// --------------------------------------------------------------------------

TEST(NaiveBatch, ZeroCandidates) {
    // n=0 — should not crash or write anything
    auto query = random_vec(16, 1);
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::L2));
    dc->compute_batch(query.data(), nullptr, 0, 16, nullptr);  // must not crash
}

TEST(NaiveBatch, SingleCandidate) {
    constexpr size_t dim = 7;  // non-multiple of 4
    auto query = random_vec(dim, 1);
    auto cand  = random_vec(dim, 2);
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::L2));
    float batch_out = 0.0f;
    dc->compute_batch(query.data(), cand.data(), 1, dim, &batch_out);
    EXPECT_FLOAT_EQ(batch_out, dc->compute(query.data(), cand.data(), dim));
}

TEST(NaiveBatch, NonMultipleOfFourDim) {
    // dim=7 exercises scalar tail in future SIMD implementations
    constexpr size_t dim = 7;
    constexpr size_t n   = 4;
    auto query = random_vec(dim, 1);
    std::vector<float> candidates(dim * n);
    for (size_t i = 0; i < n; ++i) {
        auto v = random_vec(dim, 10 + i);
        std::copy(v.begin(), v.end(), candidates.data() + i * dim);
    }
    for (Metric metric : {Metric::L2, Metric::Cosine, Metric::InnerProduct}) {
        auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(metric));
        std::vector<float> out(n);
        dc->compute_batch(query.data(), candidates.data(), n, dim, out.data());
        for (size_t i = 0; i < n; ++i) {
            float single = dc->compute(query.data(), candidates.data() + i * dim, dim);
            EXPECT_FLOAT_EQ(out[i], single);
        }
    }
}

// --------------------------------------------------------------------------
// Edge cases: zero vector, negative values
// --------------------------------------------------------------------------

TEST(NaiveCosine, ZeroVectorGuard) {
    // one vector is all zeros — denom guard should return 1.0 instead of NaN
    float a[] = {0.0f, 0.0f, 0.0f};
    float b[] = {1.0f, 2.0f, 3.0f};
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::Cosine));
    float result = dc->compute(a, b, 3);
    EXPECT_FALSE(std::isnan(result));
    EXPECT_FLOAT_EQ(result, 1.0f);
}

TEST(NaiveL2, NegativeValues) {
    float a[] = {-1.0f, -2.0f};
    float b[] = { 1.0f,  2.0f};
    // (-1-1)^2 + (-2-2)^2 = 4 + 16 = 20
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::L2));
    EXPECT_FLOAT_EQ(dc->compute(a, b, 2), 20.0f);
}

TEST(NaiveIP, NegativeDot) {
    float a[] = { 1.0f,  2.0f};
    float b[] = {-1.0f, -2.0f};
    // dot = -1 + -4 = -5 → returns 5
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::InnerProduct));
    EXPECT_FLOAT_EQ(dc->compute(a, b, 2), 5.0f);
}

// --------------------------------------------------------------------------
// Factory sanity
// --------------------------------------------------------------------------

TEST(DistanceFactory, ReturnsNonNullForAllMetrics) {
    for (Metric m : {Metric::L2, Metric::Cosine, Metric::InnerProduct}) {
        auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(m));
        EXPECT_NE(dc.get(), nullptr) << "metric=" << static_cast<int>(m);
    }
}

TEST(DistanceFactory, CreateScalarReturnsNonNullForAllMetrics) {
    for (Metric m : {Metric::L2, Metric::Cosine, Metric::InnerProduct}) {
        auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create_scalar(m));
        EXPECT_NE(dc.get(), nullptr) << "metric=" << static_cast<int>(m);
    }
}

}  // namespace
}  // namespace vectordb
