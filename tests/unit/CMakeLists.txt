function(vdb_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE
        vortex_core vortex_storage vortex_server GTest::gtest_main)
    gtest_discover_tests(${name})
endfunction()

vdb_test(test_smoke)
vdb_test(test_distance)

if(VECTORDB_ARCH STREQUAL "arm")
    vdb_test(test_distance_neon)   # Day 3: NEON vs scalar correctness
elseif(VECTORDB_ARCH STREQUAL "x86")
    vdb_test(test_distance_avx2)   # Day 4: AVX2 vs scalar correctness + dispatch
endif()
