// -----------------------------------------------------------------------------
// bench_distance.cpp — scalar vs NEON distance throughput benchmark  (Day 3)
//
// Build and run:
//   cmake -B build -DVECTORDB_BENCH=ON -DCMAKE_BUILD_TYPE=Release
//   cmake --build build --target bench_distance
//   ./build/bench/bench_distance
//
// Reports throughput in ns/op for each metric × dim combination.
// Compare BM_Scalar* vs BM_Simd* rows to see NEON speedup.
// -----------------------------------------------------------------------------
#include <benchmark/benchmark.h>
#include "core/distance.h"
#include <random>
#include <vector>

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
// Single compute() call benchmarks
// Template parameter: Metric
// Benchmark parameter: dim (128, 768, 1536)
// --------------------------------------------------------------------------

template<Metric M>
void BM_Scalar(benchmark::State& state) {
    size_t dim = state.range(0);
    auto a = random_vec(dim, 1);
    auto b = random_vec(dim, 2);
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create_scalar(M));
    for (auto _ : state) {
        benchmark::DoNotOptimize(dc->compute(a.data(), b.data(), dim));
    }
    state.SetItemsProcessed(state.iterations() * dim);
}

template<Metric M>
void BM_Simd(benchmark::State& state) {
    size_t dim = state.range(0);
    auto a = random_vec(dim, 1);
    auto b = random_vec(dim, 2);
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(M));
    for (auto _ : state) {
        benchmark::DoNotOptimize(dc->compute(a.data(), b.data(), dim));
    }
    state.SetItemsProcessed(state.iterations() * dim);
}

BENCHMARK_TEMPLATE(BM_Scalar, Metric::L2)          ->Arg(128)->Arg(768)->Arg(1536);
BENCHMARK_TEMPLATE(BM_Simd,   Metric::L2)          ->Arg(128)->Arg(768)->Arg(1536);
BENCHMARK_TEMPLATE(BM_Scalar, Metric::Cosine)      ->Arg(128)->Arg(768)->Arg(1536);
BENCHMARK_TEMPLATE(BM_Simd,   Metric::Cosine)      ->Arg(128)->Arg(768)->Arg(1536);
BENCHMARK_TEMPLATE(BM_Scalar, Metric::InnerProduct)->Arg(128)->Arg(768)->Arg(1536);
BENCHMARK_TEMPLATE(BM_Simd,   Metric::InnerProduct)->Arg(128)->Arg(768)->Arg(1536);

// --------------------------------------------------------------------------
// compute_batch benchmarks: 1024 candidates, dim=768
// --------------------------------------------------------------------------

template<Metric M>
void BM_ScalarBatch(benchmark::State& state) {
    constexpr size_t n = 1024;
    size_t dim = state.range(0);
    auto query = random_vec(dim, 1);
    std::vector<float> candidates(dim * n);
    for (size_t i = 0; i < n; ++i) {
        auto v = random_vec(dim, 100 + i);
        std::copy(v.begin(), v.end(), candidates.data() + i * dim);
    }
    std::vector<float> out(n);
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create_scalar(M));
    for (auto _ : state) {
        dc->compute_batch(query.data(), candidates.data(), n, dim, out.data());
        benchmark::DoNotOptimize(out.data());
    }
    state.SetItemsProcessed(state.iterations() * n * dim);
}

template<Metric M>
void BM_SimdBatch(benchmark::State& state) {
    constexpr size_t n = 1024;
    size_t dim = state.range(0);
    auto query = random_vec(dim, 1);
    std::vector<float> candidates(dim * n);
    for (size_t i = 0; i < n; ++i) {
        auto v = random_vec(dim, 100 + i);
        std::copy(v.begin(), v.end(), candidates.data() + i * dim);
    }
    std::vector<float> out(n);
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(M));
    for (auto _ : state) {
        dc->compute_batch(query.data(), candidates.data(), n, dim, out.data());
        benchmark::DoNotOptimize(out.data());
    }
    state.SetItemsProcessed(state.iterations() * n * dim);
}

BENCHMARK_TEMPLATE(BM_ScalarBatch, Metric::L2)          ->Arg(768);
BENCHMARK_TEMPLATE(BM_SimdBatch,   Metric::L2)          ->Arg(768);
BENCHMARK_TEMPLATE(BM_ScalarBatch, Metric::Cosine)      ->Arg(768);
BENCHMARK_TEMPLATE(BM_SimdBatch,   Metric::Cosine)      ->Arg(768);
BENCHMARK_TEMPLATE(BM_ScalarBatch, Metric::InnerProduct)->Arg(768);
BENCHMARK_TEMPLATE(BM_SimdBatch,   Metric::InnerProduct)->Arg(768);

// --------------------------------------------------------------------------
// Scalar tail: dims not divisible by 4
// Shows overhead of the tail loop when NEON can't fill a full 4-wide batch.
// dim=769 = 192 full NEON iterations + 1 scalar tail element.
// dim=7   = 1  full NEON iteration  + 3 scalar tail elements (heavy tail).
// --------------------------------------------------------------------------

BENCHMARK_TEMPLATE(BM_Scalar, Metric::L2)->Arg(7)->Arg(769);
BENCHMARK_TEMPLATE(BM_Simd,   Metric::L2)->Arg(7)->Arg(769);

// --------------------------------------------------------------------------
// Unaligned memory: offset pointer by 1 float (4 bytes) into the buffer.
// std::vector guarantees 16-byte alignment for the base pointer; shifting
// by 4 bytes breaks that alignment. vld1q_f32 on ARM supports unaligned
// loads, but this measures whether there is any penalty in practice.
// --------------------------------------------------------------------------

template<Metric M>
void BM_SimdUnaligned(benchmark::State& state) {
    size_t dim = state.range(0);
    // Allocate dim+1 floats so we can offset by 1 without out-of-bounds.
    std::vector<float> buf_a(dim + 1), buf_b(dim + 1);
    auto rng = std::mt19937(42);
    auto dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
    for (auto& x : buf_a) x = dist(rng);
    for (auto& x : buf_b) x = dist(rng);
    // Offset by 1 float to break natural 16-byte alignment.
    const float* a = buf_a.data() + 1;
    const float* b = buf_b.data() + 1;
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(M));
    for (auto _ : state) {
        benchmark::DoNotOptimize(dc->compute(a, b, dim));
    }
    state.SetItemsProcessed(state.iterations() * dim);
}

BENCHMARK_TEMPLATE(BM_SimdUnaligned, Metric::L2)          ->Arg(128)->Arg(768)->Arg(1536);
BENCHMARK_TEMPLATE(BM_SimdUnaligned, Metric::Cosine)      ->Arg(128)->Arg(768)->Arg(1536);
BENCHMARK_TEMPLATE(BM_SimdUnaligned, Metric::InnerProduct)->Arg(128)->Arg(768)->Arg(1536);

}  // namespace
}  // namespace vectordb

BENCHMARK_MAIN();
