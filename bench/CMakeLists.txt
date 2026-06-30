# Google Benchmark dependency added Week 6 (Day 25).
# Benchmarks are not compiled by default in the regular build.
#
# To enable:
#   cmake -B build -DVECTORDB_BENCH=ON
#   cmake --build build --target bench_distance bench_hnsw bench_storage

option(VECTORDB_BENCH "Build benchmarks (requires Google Benchmark)" OFF)
option(VORTEXDB_NO_PREFETCH "Disable __builtin_prefetch (look-ahead graph + compute_batch) for profiling" OFF)

if(VECTORDB_BENCH)
    foreach(bench distance hnsw storage)
        add_executable(bench_${bench} bench_${bench}.cpp)
        target_link_libraries(bench_${bench} PRIVATE
            vortex_core vortex_storage vortex_server
            benchmark::benchmark)
    endforeach()
endif()

# bench_profile: standalone profiling harness, no Google Benchmark dependency.
# Build:                cmake -B build -DVECTORDB_PROFILE=ON
# Disable prefetch:     cmake -B build -DVECTORDB_PROFILE=ON -DVORTEXDB_NO_PREFETCH=ON
option(VECTORDB_PROFILE "Build profiling harness (bench_profile)" OFF)

if(VECTORDB_PROFILE)
    add_executable(bench_profile bench_profile.cpp)
    target_include_directories(bench_profile PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(bench_profile PRIVATE vortex_core vortex_storage vortex_server)
    if(VORTEXDB_NO_PREFETCH)
        target_compile_definitions(bench_profile PRIVATE VORTEXDB_NO_PREFETCH)
        target_compile_definitions(vortex_core PRIVATE VORTEXDB_NO_PREFETCH)
    endif()
endif()
