// bench_profile.cpp — SIMD profiling harness
//
// Measures time breakdown during HNSW search:
//   - Distance compute fraction vs graph traversal overhead
//   - Prefetch on vs off comparison (build with -DVORTEXDB_NO_PREFETCH to disable)
//
// Usage (build with VECTORDB_BENCH=ON):
//   ./build/bench/bench_profile [--dim 128] [--n 100000] [--queries 10000] [--ef 200]
//
// For macOS profiling:
//   sample ./build/bench/bench_profile 10 -file profile.txt
//
// For perf on Linux:
//   perf stat -e cache-misses,cache-references,instructions,cycles ./build/bench/bench_profile

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/distance.h"
#include "core/hnsw_index.h"

using namespace vectordb;
using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static std::vector<float> random_vec(size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

int main(int argc, char** argv) {
    size_t dim     = 128;
    size_t n       = 100'000;
    size_t n_query = 10'000;
    int    ef      = 200;
    int    top_k   = 10;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dim"     && i+1 < argc) dim     = std::stoul(argv[++i]);
        if (arg == "--n"       && i+1 < argc) n       = std::stoul(argv[++i]);
        if (arg == "--queries" && i+1 < argc) n_query = std::stoul(argv[++i]);
        if (arg == "--ef"      && i+1 < argc) ef      = std::stoi(argv[++i]);
    }

#ifdef VORTEXDB_NO_PREFETCH
    const char* prefetch_mode = "OFF";
#else
    const char* prefetch_mode = "ON";
#endif

    printf("=== VortexDB SIMD Profiling Harness ===\n");
    printf("dim=%zu  n=%zu  queries=%zu  ef=%d  prefetch=%s\n\n",
           dim, n, n_query, ef, prefetch_mode);

    std::mt19937 rng(42);

    // ------------------------------------------------------------------
    // Build index
    // ------------------------------------------------------------------
    HnswConfig cfg;
    cfg.dim    = dim;
    cfg.metric = Metric::L2;

    auto index = std::make_unique<HnswIndex>(cfg);

    std::vector<std::vector<float>> base(n);
    printf("Inserting %zu vectors...\n", n);
    auto t_insert = Clock::now();
    for (size_t i = 0; i < n; ++i) {
        base[i] = random_vec(dim, rng);
        index->insert(base[i].data());
    }
    double insert_ms = elapsed_ms(t_insert);
    printf("  Done: %.1f ms  (%.0f vec/s)\n\n", insert_ms, n / (insert_ms / 1000.0));

    // ------------------------------------------------------------------
    // Search — measure total time
    // ------------------------------------------------------------------
    std::vector<std::vector<float>> queries(n_query);
    for (auto& q : queries) q = random_vec(dim, rng);

    printf("Searching %zu queries (ef=%d, top_k=%d)...\n", n_query, ef, top_k);

    // Warmup
    for (size_t i = 0; i < 100; ++i)
        index->search(queries[i % n_query].data(), top_k, ef);

    auto t_search = Clock::now();
    size_t total_results = 0;
    for (size_t i = 0; i < n_query; ++i) {
        auto res = index->search(queries[i].data(), top_k, ef);
        total_results += res.size();
    }
    double search_ms = elapsed_ms(t_search);

    double qps = n_query / (search_ms / 1000.0);
    double ms_per_query = search_ms / n_query;

    printf("  Total:       %.1f ms\n", search_ms);
    printf("  Per query:   %.3f ms\n", ms_per_query);
    printf("  QPS:         %.0f\n", qps);
    printf("  Results:     %zu\n\n", total_results);

    // ------------------------------------------------------------------
    // Distance compute microbenchmark (isolate compute_batch cost)
    // ------------------------------------------------------------------
    printf("Distance compute microbenchmark (compute_batch, %zu vectors)...\n", n);
    auto dc = std::unique_ptr<DistanceCompute>(DistanceCompute::create(Metric::L2));

    // Flatten base into contiguous array for cache-realistic test
    std::vector<float> flat(n * dim);
    for (size_t i = 0; i < n; ++i)
        std::memcpy(flat.data() + i * dim, base[i].data(), dim * sizeof(float));

    std::vector<float> dist_out(n);
    const size_t kDistBatches = 5;
    auto t_dist = Clock::now();
    for (size_t b = 0; b < kDistBatches; ++b) {
        const float* q = queries[b % n_query].data();
        dc->compute_batch(q, flat.data(), n, dim, dist_out.data());
    }
    double dist_ms = elapsed_ms(t_dist) / kDistBatches;
    double dist_ns_per_vec = dist_ms * 1e6 / n;

    printf("  compute_batch over %zu vecs: %.2f ms  (%.1f ns/vec)\n\n",
           n, dist_ms, dist_ns_per_vec);

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    // Estimate: how much of search time is distance compute?
    // Each query visits ~ef*log(n) nodes. ef=200, n=100K → ~200*17 ≈ 3400 nodes.
    double nodes_per_query = ef * std::log2(static_cast<double>(n));
    double dist_cost_per_query_ms = dist_ns_per_vec * nodes_per_query / 1e6;
    double dist_fraction = dist_cost_per_query_ms / ms_per_query * 100.0;

    printf("=== Estimated breakdown ===\n");
    printf("  Nodes visited per query (est): %.0f\n", nodes_per_query);
    printf("  Distance cost per query (est): %.3f ms\n", dist_cost_per_query_ms);
    printf("  Actual time per query:         %.3f ms\n", ms_per_query);
    printf("  Estimated distance fraction:   %.0f%%\n", dist_fraction);
    printf("  Graph overhead fraction:       %.0f%%\n\n", 100.0 - dist_fraction);

    printf("(Run with -DVORTEXDB_NO_PREFETCH to compare prefetch on vs off)\n");
    return 0;
}
