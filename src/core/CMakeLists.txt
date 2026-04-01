set(CORE_SOURCES
    distance_naive.cpp   # Day 2 — scalar L2 / Cosine / IP + factory
    collection.cpp       # trivial schema container, complete
)

if(VECTORDB_ARCH STREQUAL "arm")
    list(APPEND CORE_SOURCES distance_neon.cpp)      # Day 3
elseif(VECTORDB_ARCH STREQUAL "x86")
    list(APPEND CORE_SOURCES distance_avx2.cpp)  # Day 4
endif()

# hnsw_index.cpp    added Day 6–10
# arena_allocator.cpp added Day 13

add_library(vortex_core STATIC ${CORE_SOURCES})
target_include_directories(vortex_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
