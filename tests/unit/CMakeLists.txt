function(vdb_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE
        vortex_core vortex_storage vortex_server GTest::gtest_main)
    gtest_discover_tests(${name})
endfunction()

vdb_test(test_smoke)
vdb_test(test_distance)

vdb_test(test_hnsw_ut)   # Day 6-8: HNSW insert/search/remove/recall/graph structure/pruning
vdb_test(test_wal_ut)         # Day 11-12: WAL append/sync/iterate/CRC/reopen/truncate/replay
vdb_test(test_vector_file_ut) # Day 13: mmap vector storage append/read/grow/persist

# Day 9: SIFT recall test (skipped if data not present)
add_executable(test_sift_recall test_sift_recall.cpp)
target_link_libraries(test_sift_recall PRIVATE
    vortex_core vortex_storage vortex_server vortex_tools GTest::gtest_main)
gtest_discover_tests(test_sift_recall)

if(VECTORDB_ARCH STREQUAL "arm")
    vdb_test(test_distance_neon)   # Day 3: NEON vs scalar correctness
elseif(VECTORDB_ARCH STREQUAL "x86")
    vdb_test(test_distance_avx2)   # Day 4: AVX2 vs scalar correctness + dispatch
endif()
