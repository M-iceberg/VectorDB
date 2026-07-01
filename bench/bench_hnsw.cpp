// HNSW microbenchmarks via Google Benchmark.
// End-to-end QPS and recall measurements are in bench/bench_profile.cpp
// and the Python scripts bench/bench_sift.py, bench/bench_glove.py.
#include <benchmark/benchmark.h>
#include "core/hnsw_index.h"

BENCHMARK_MAIN();
