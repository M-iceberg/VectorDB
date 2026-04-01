// -----------------------------------------------------------------------------
// bench_distance.cpp — scalar vs SIMD distance throughput benchmark
//
// Build and run:
//   cmake -B build -DVECTORDB_BENCH=ON -DCMAKE_BUILD_TYPE=Release
//   cmake --build build --target bench_distance
//   ./build/bench/bench_distance
//
// Output columns:
//   Time (ns/op)      — nanoseconds per compute() call; lower is faster.
//   items/per_second  — floats processed per second (dim elements per call);
//                       higher is faster. Useful for comparing across dims.
//
// Example (ARM, dim=768):
//   BM_Scalar<L2>/768    310 ns    2.47G items/s
//   BM_Simd<L2>/768      109 ns    7.03G items/s   → 2.8x speedup
//
// Benchmark cases:
//   BM_Scalar / BM_Simd          — scalar vs SIMD single compute(), 3 metrics × dim 128/768/1536
//   BM_ScalarBatch / BM_SimdBatch— scalar vs SIMD compute_batch(), 1024 candidates, dim=768
//   BM_BatchNoPrefetch           — compute_batch without prefetch, compared against BM_SimdBatch
//                                  to isolate the effect of __builtin_prefetch
//   BM_Avx2 / BM_Avx2Batch      — explicit AVX2 single and batch (x86 only)
//   BM_Scalar / BM_Simd (Avx2)  — scalar tail: dim=7 and dim=769 (non-multiples of 4/8)
//   BM_SimdUnaligned             — SIMD with unaligned pointer, 3 metrics × dim 128/768/1536
// -----------------------------------------------------------------------------
#include <benchmark/benchmark.h>
#include "core/distance.h"
#include <memory>
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
// Prefetch effect: compute_batch with vs without prefetch
// BM_BatchNoPrefetch manually loops without prefetch to isolate the effect.
// BM_SimdBatch uses compute_batch() which has prefetch built in.
// Run with large n (1024) and large dim so candidates don't fit in L1 cache,
// making prefetch matter. If they fit in cache, both will look the same.
// --------------------------------------------------------------------------

template<Metric M>
void BM_BatchNoPrefetch(benchmark::State& state) {
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
        // Same loop as compute_batch but without __builtin_prefetch.
        for (size_t i = 0; i < n; ++i) {
            out[i] = dc->compute(query.data(), candidates.data() + i * dim, dim);
        }
        benchmark::DoNotOptimize(out.data());
    }
    state.SetItemsProcessed(state.iterations() * n * dim);
}

BENCHMARK_TEMPLATE(BM_BatchNoPrefetch, Metric::L2)          ->Arg(768);
BENCHMARK_TEMPLATE(BM_SimdBatch,       Metric::L2)          ->Arg(768);
BENCHMARK_TEMPLATE(BM_BatchNoPrefetch, Metric::Cosine)      ->Arg(768);
BENCHMARK_TEMPLATE(BM_SimdBatch,       Metric::Cosine)      ->Arg(768);
BENCHMARK_TEMPLATE(BM_BatchNoPrefetch, Metric::InnerProduct)->Arg(768);
BENCHMARK_TEMPLATE(BM_SimdBatch,       Metric::InnerProduct)->Arg(768);

// --------------------------------------------------------------------------
// x86 AVX2 explicit benchmarks — compiled only on x86_64
// BM_Simd already calls create() which returns AVX2 on x86; these use
// create_avx2() directly so the AVX2 label is unambiguous in CI output.
// --------------------------------------------------------------------------

#if defined(VECTORDB_ARCH_X86)

template<Metric M>
void BM_Avx2(benchmark::State& state) {
    size_t dim = state.range(0);
    auto a = random_vec(dim, 1);
    auto b = random_vec(dim, 2);
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create_avx2(M));
    for (auto _ : state) {
        benchmark::DoNotOptimize(dc->compute(a.data(), b.data(), dim));
    }
    state.SetItemsProcessed(state.iterations() * dim);
}

template<Metric M>
void BM_Avx2Batch(benchmark::State& state) {
    constexpr size_t n = 1024;
    size_t dim = state.range(0);
    auto query = random_vec(dim, 1);
    std::vector<float> candidates(dim * n);
    for (size_t i = 0; i < n; ++i) {
        auto v = random_vec(dim, 100 + i);
        std::copy(v.begin(), v.end(), candidates.data() + i * dim);
    }
    std::vector<float> out(n);
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create_avx2(M));
    for (auto _ : state) {
        dc->compute_batch(query.data(), candidates.data(), n, dim, out.data());
        benchmark::DoNotOptimize(out.data());
    }
    state.SetItemsProcessed(state.iterations() * n * dim);
}

BENCHMARK_TEMPLATE(BM_Avx2, Metric::L2)          ->Arg(128)->Arg(768)->Arg(1536);
BENCHMARK_TEMPLATE(BM_Avx2, Metric::Cosine)      ->Arg(128)->Arg(768)->Arg(1536);
BENCHMARK_TEMPLATE(BM_Avx2, Metric::InnerProduct)->Arg(128)->Arg(768)->Arg(1536);
BENCHMARK_TEMPLATE(BM_Avx2Batch, Metric::L2)          ->Arg(768);
BENCHMARK_TEMPLATE(BM_Avx2Batch, Metric::Cosine)      ->Arg(768);
BENCHMARK_TEMPLATE(BM_Avx2Batch, Metric::InnerProduct)->Arg(768);

// Scalar tail on x86: dims not divisible by 8
// dim=769 = 96 full AVX2 iterations + 1 scalar tail element.
// dim=7   = 0 full AVX2 iterations  + 7 scalar tail elements (all tail).
BENCHMARK_TEMPLATE(BM_Scalar, Metric::L2)->Arg(7)->Arg(769);
BENCHMARK_TEMPLATE(BM_Avx2,  Metric::L2)->Arg(7)->Arg(769);

#endif  // VECTORDB_ARCH_X86

// --------------------------------------------------------------------------
// Scalar tail: dims not divisible by 4 (NEON / ARM)
// Shows overhead of the tail loop when NEON can't fill a full 4-wide batch.
// dim=769 = 192 full NEON iterations + 1 scalar tail element.
// dim=7   = 1  full NEON iteration  + 3 scalar tail elements (heavy tail).
// --------------------------------------------------------------------------

#if defined(VECTORDB_ARCH_ARM)
BENCHMARK_TEMPLATE(BM_Scalar, Metric::L2)->Arg(7)->Arg(769);
BENCHMARK_TEMPLATE(BM_Simd,   Metric::L2)->Arg(7)->Arg(769);
#endif  // VECTORDB_ARCH_ARM

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
