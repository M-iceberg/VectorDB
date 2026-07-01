set(CORE_SOURCES
    distance_naive.cpp   # Day 2 — scalar L2 / Cosine / IP + factory
    collection.cpp       # trivial schema container, complete
)

if(VECTORDB_ARCH STREQUAL "arm")
    list(APPEND CORE_SOURCES distance_neon.cpp)      # Day 3
elseif(VECTORDB_ARCH STREQUAL "x86")
    list(APPEND CORE_SOURCES distance_avx2.cpp)  # Day 4
endif()

list(APPEND CORE_SOURCES hnsw_index.cpp)

add_library(vortex_core STATIC ${CORE_SOURCES})
target_include_directories(vortex_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
