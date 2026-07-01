// Storage microbenchmarks via Google Benchmark.
// WAL throughput and recovery time measurements are in bench/bench_recovery.py.
#include <benchmark/benchmark.h>
#include "storage/wal.h"
#include "storage/vector_file.h"

BENCHMARK_MAIN();
