cmake_minimum_required(VERSION 3.20)
project(vectordb VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --------------------------------------------------------------------------
# SIMD platform detection (sets VECTORDB_ARCH, adds compile flags/defs)
# --------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(SimdDetect)

# --------------------------------------------------------------------------
# VECTORDB_PYTHON_ONLY: skip tests/bench/tools (used by pip install)
# --------------------------------------------------------------------------
option(VECTORDB_PYTHON_ONLY "Skip tests, bench, and tools (for pip install)" OFF)

include(FetchContent)

if(NOT VECTORDB_PYTHON_ONLY)
    # --------------------------------------------------------------------------
    # Google Test
    # --------------------------------------------------------------------------
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # --------------------------------------------------------------------------
    # Google Benchmark (only fetched when VECTORDB_BENCH=ON)
    # --------------------------------------------------------------------------
    if(VECTORDB_BENCH)
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            googlebenchmark
            URL https://github.com/google/benchmark/archive/refs/tags/v1.8.4.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(googlebenchmark)
    endif()

    enable_testing()
    include(GoogleTest)
endif()

# --------------------------------------------------------------------------
# Python bindings (opt-in: cmake -DVECTORDB_PYTHON=ON)
# --------------------------------------------------------------------------
option(VECTORDB_PYTHON "Build Python bindings via pybind11" OFF)
if(VECTORDB_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
    FetchContent_Declare(
        pybind11
        URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# --------------------------------------------------------------------------
# Subdirectories
# --------------------------------------------------------------------------
add_subdirectory(src)
if(NOT VECTORDB_PYTHON_ONLY)
    add_subdirectory(tests)
    add_subdirectory(bench)
    add_subdirectory(tools)
endif()
if(VECTORDB_PYTHON)
    add_subdirectory(python)
endif()
