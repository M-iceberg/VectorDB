#!/usr/bin/env bash
# End-to-end benchmark script — used Week 6.
# Usage: ./tools/run_benchmark.sh <build_dir> <data_dir>
set -euo pipefail

BUILD_DIR="${1:-build}"
DATA_DIR="${2:-data}"

echo "=== Distance benchmark ==="
"${BUILD_DIR}/bench/bench_distance" --benchmark_format=json

echo "=== HNSW benchmark (SIFT-1M) ==="
"${BUILD_DIR}/bench/bench_hnsw" --benchmark_format=json \
    --sift_path="${DATA_DIR}/sift"

echo "=== Storage benchmark ==="
"${BUILD_DIR}/bench/bench_storage" --benchmark_format=json
