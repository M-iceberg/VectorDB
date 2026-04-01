#include <gtest/gtest.h>
#include "core/distance.h"
#include <cmath>
#include <memory>
#include <random>
#include <vector>

// Day 3: NEON vs scalar correctness tests — compiled only on aarch64.
// Validates that create() (NEON) and create_scalar() (naive) agree on all dims.
// Non-multiples of 4 (1, 3, 7) specifically exercise the scalar tail loop.

namespace vectordb {
namespace {

std::vector<float> random_vec(size_t dim, uint64_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

// --------------------------------------------------------------------------
// NEON vs scalar: create() must agree with create_scalar() on all dims
// --------------------------------------------------------------------------

class NeonVsScalar : public ::testing::TestWithParam<size_t> {};

TEST_P(NeonVsScalar, L2AgreeWithScalar) {
    size_t dim = GetParam();
    auto a = random_vec(dim, 7);
    auto b = random_vec(dim, 13);
    auto neon   = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::L2));
    auto scalar = std::unique_ptr<DistanceCompute>(DistanceCompute::create_scalar(Metric::L2));
    float got = neon->compute(a.data(), b.data(), dim);
    float ref = scalar->compute(a.data(), b.data(), dim);
    EXPECT_NEAR(got, ref, ref * 1e-5f + 1e-6f);
}

TEST_P(NeonVsScalar, CosineAgreeWithScalar) {
    size_t dim = GetParam();
    auto a = random_vec(dim, 7);
    auto b = random_vec(dim, 13);
    auto neon   = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::Cosine));
    auto scalar = std::unique_ptr<DistanceCompute>(DistanceCompute::create_scalar(Metric::Cosine));
    float got = neon->compute(a.data(), b.data(), dim);
    float ref = scalar->compute(a.data(), b.data(), dim);
    EXPECT_NEAR(got, ref, 1e-5f);
}

TEST_P(NeonVsScalar, IPAgreeWithScalar) {
    size_t dim = GetParam();
    auto a = random_vec(dim, 7);
    auto b = random_vec(dim, 13);
    auto neon   = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::InnerProduct));
    auto scalar = std::unique_ptr<DistanceCompute>(DistanceCompute::create_scalar(Metric::InnerProduct));
    float got = neon->compute(a.data(), b.data(), dim);
    float ref = scalar->compute(a.data(), b.data(), dim);
    EXPECT_NEAR(got, ref, std::abs(ref) * 1e-5f + 1e-6f);
}

// Same dims as the scalar random suite; non-multiples of 4 exercise tail handling.
INSTANTIATE_TEST_SUITE_P(Dims, NeonVsScalar,
    ::testing::Values(1, 3, 4, 7, 16, 128, 768, 1536));

// --------------------------------------------------------------------------
// compute_batch: NEON batch results must match NEON single-call results
// --------------------------------------------------------------------------

TEST(NeonBatch, MatchesSingleCalls) {
    constexpr size_t dim = 64;
    constexpr size_t n   = 8;
    auto query = random_vec(dim, 5);
    std::vector<float> candidates(dim * n);
    for (size_t i = 0; i < n; ++i) {
        auto v = random_vec(dim, 200 + i);
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

}  // namespace
}  // namespace vectordb
