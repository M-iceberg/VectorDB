# Google Benchmark dependency added Week 6 (Day 25).
# Benchmarks are not compiled by default in the regular build.
#
# To enable:
#   cmake -B build -DVECTORDB_BENCH=ON
#   cmake --build build --target bench_distance bench_hnsw bench_storage

option(VECTORDB_BENCH "Build benchmarks (requires Google Benchmark)" OFF)

if(VECTORDB_BENCH)
    find_package(benchmark REQUIRED)

    foreach(bench distance hnsw storage)
        add_executable(bench_${bench} bench_${bench}.cpp)
        target_link_libraries(bench_${bench} PRIVATE
            vortex_core vortex_storage vortex_server
            benchmark::benchmark)
    endforeach()
endif()
