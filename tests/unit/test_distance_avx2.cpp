#include <gtest/gtest.h>
#include "core/distance.h"
#include <cmath>
#include <memory>
#include <random>
#include <vector>

// Day 4: AVX2 vs scalar correctness tests — compiled only on x86_64.
// Validates that create_avx2() and create_scalar() agree on all dims.
// Non-multiples of 8 (1, 3, 7, 9) specifically exercise the scalar tail loop.

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
// AVX2 vs scalar: create_avx2() must agree with create_scalar() on all dims
// --------------------------------------------------------------------------

class Avx2VsScalar : public ::testing::TestWithParam<size_t> {};

TEST_P(Avx2VsScalar, L2AgreeWithScalar) {
    size_t dim = GetParam();
    auto a = random_vec(dim, 7);
    auto b = random_vec(dim, 13);
    auto avx2   = std::unique_ptr<DistanceCompute>(DistanceCompute::create_avx2(Metric::L2));
    auto scalar = std::unique_ptr<DistanceCompute>(DistanceCompute::create_scalar(Metric::L2));
    float got = avx2->compute(a.data(), b.data(), dim);
    float ref = scalar->compute(a.data(), b.data(), dim);
    EXPECT_NEAR(got, ref, ref * 1e-5f + 1e-6f);
}

TEST_P(Avx2VsScalar, CosineAgreeWithScalar) {
    size_t dim = GetParam();
    auto a = random_vec(dim, 7);
    auto b = random_vec(dim, 13);
    auto avx2   = std::unique_ptr<DistanceCompute>(DistanceCompute::create_avx2(Metric::Cosine));
    auto scalar = std::unique_ptr<DistanceCompute>(DistanceCompute::create_scalar(Metric::Cosine));
    float got = avx2->compute(a.data(), b.data(), dim);
    float ref = scalar->compute(a.data(), b.data(), dim);
    EXPECT_NEAR(got, ref, 1e-5f);
}

TEST_P(Avx2VsScalar, IPAgreeWithScalar) {
    size_t dim = GetParam();
    auto a = random_vec(dim, 7);
    auto b = random_vec(dim, 13);
    auto avx2   = std::unique_ptr<DistanceCompute>(DistanceCompute::create_avx2(Metric::InnerProduct));
    auto scalar = std::unique_ptr<DistanceCompute>(DistanceCompute::create_scalar(Metric::InnerProduct));
    float got = avx2->compute(a.data(), b.data(), dim);
    float ref = scalar->compute(a.data(), b.data(), dim);
    EXPECT_NEAR(got, ref, std::abs(ref) * 1e-5f + 1e-6f);
}

// Dims include non-multiples of 8 to exercise scalar tail handling.
// dim=1,3,7: nearly all work in tail; dim=9: 1 AVX2 iter + 1 tail element.
INSTANTIATE_TEST_SUITE_P(Dims, Avx2VsScalar,
    ::testing::Values(1, 3, 7, 8, 9, 16, 128, 768, 1536));

// --------------------------------------------------------------------------
// Runtime dispatch: create() must return AVX2 on AVX2-capable hardware
// --------------------------------------------------------------------------

TEST(Avx2Dispatch, CreateReturnsAvx2WhenSupported) {
    // If this test runs, we're on x86. Check that create() and create_avx2()
    // return the same results — confirming dispatch selected AVX2.
    if (!__builtin_cpu_supports("avx2")) {
        GTEST_SKIP() << "AVX2 not supported on this CPU, skipping dispatch test";
    }
    constexpr size_t dim = 128;
    auto a = random_vec(dim, 42);
    auto b = random_vec(dim, 99);
    for (Metric m : {Metric::L2, Metric::Cosine, Metric::InnerProduct}) {
        auto dispatched = std::unique_ptr<DistanceCompute>(DistanceCompute::create(m));
        auto avx2       = std::unique_ptr<DistanceCompute>(DistanceCompute::create_avx2(m));
        EXPECT_FLOAT_EQ(dispatched->compute(a.data(), b.data(), dim),
                        avx2->compute(a.data(), b.data(), dim))
            << "metric=" << static_cast<int>(m);
    }
}

// --------------------------------------------------------------------------
// compute_batch: AVX2 batch results must match AVX2 single-call results
// --------------------------------------------------------------------------

TEST(Avx2Batch, MatchesSingleCalls) {
    constexpr size_t dim = 64;
    constexpr size_t n   = 8;
    auto query = random_vec(dim, 5);
    std::vector<float> candidates(dim * n);
    for (size_t i = 0; i < n; ++i) {
        auto v = random_vec(dim, 200 + i);
        std::copy(v.begin(), v.end(), candidates.data() + i * dim);
    }
    for (Metric metric : {Metric::L2, Metric::Cosine, Metric::InnerProduct}) {
        auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create_avx2(metric));
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
// hsum256 correctness — tested indirectly via known exact values
// hsum256 is in an anonymous namespace so can't be called directly;
// we use dim=8 with hand-crafted vectors so the result is known exactly
// and must survive the full 8-lane horizontal sum without rounding error.
// --------------------------------------------------------------------------

TEST(Avx2Hsum, ExactSumAllEightLanes) {
    // a = [1,0,0,0,0,0,0,0], b = [0,0,0,0,0,0,0,0]
    // L2 = (1-0)^2 + 0*7 = 1.0 — exactly one lane contributes, rest are 0
    float a[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float b[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create_avx2(Metric::L2));
    EXPECT_FLOAT_EQ(dc->compute(a, b, 8), 1.0f);
}

TEST(Avx2Hsum, ExactSumAllLanesContribute) {
    // a = [1,1,1,1,1,1,1,1], b = [0,0,0,0,0,0,0,0]
    // L2 = 8 * 1^2 = 8.0 — all 8 lanes contribute equally, tests full reduction
    float a[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float b[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create_avx2(Metric::L2));
    EXPECT_FLOAT_EQ(dc->compute(a, b, 8), 8.0f);
}

// --------------------------------------------------------------------------
// Fallback path: when AVX2 is not available, create() must return scalar results.
// The no-AVX2 code path cannot be triggered on AVX2-capable hardware without
// mocking __builtin_cpu_supports. Instead we verify that create_scalar() —
// which is what create() falls back to — produces correct results on x86,
// ensuring the fallback implementation itself is sound.
// --------------------------------------------------------------------------

TEST(Avx2Fallback, ScalarFallbackCorrect) {
    constexpr size_t dim = 128;
    auto a = random_vec(dim, 42);
    auto b = random_vec(dim, 99);
    for (Metric m : {Metric::L2, Metric::Cosine, Metric::InnerProduct}) {
        auto scalar = std::unique_ptr<DistanceCompute>(DistanceCompute::create_scalar(m));
        auto avx2   = std::unique_ptr<DistanceCompute>(DistanceCompute::create_avx2(m));
        float scalar_result = scalar->compute(a.data(), b.data(), dim);
        float avx2_result   = avx2->compute(a.data(), b.data(), dim);
        EXPECT_NEAR(scalar_result, avx2_result, std::abs(avx2_result) * 1e-5f + 1e-6f)
            << "metric=" << static_cast<int>(m);
    }
}

}  // namespace
}  // namespace vectordb
