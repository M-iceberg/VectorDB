// Storage benchmark: WAL throughput, mmap read latency — added Week 6.
#include <benchmark/benchmark.h>
#include "storage/wal.h"
#include "storage/vector_file.h"

// TODO Day 28: BM_WalAppend, BM_VectorFileRead

BENCHMARK_MAIN();
