set(CORE_SOURCES
    distance_naive.cpp
    hnsw_index.cpp
    arena_allocator.cpp
    collection.cpp
)

if(VECTORDB_ARCH STREQUAL "arm")
    list(APPEND CORE_SOURCES distance_neon.cpp)
elseif(VECTORDB_ARCH STREQUAL "x86")
    list(APPEND CORE_SOURCES distance_avx2.cpp distance_dispatch.cpp)
endif()

add_library(vortex_core STATIC ${CORE_SOURCES})
target_include_directories(vortex_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
