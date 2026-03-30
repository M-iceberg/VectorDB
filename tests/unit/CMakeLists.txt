function(vdb_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE
        vortex_core vortex_storage vortex_server GTest::gtest_main)
    gtest_discover_tests(${name})
endfunction()

vdb_test(test_smoke)
vdb_test(test_distance)
