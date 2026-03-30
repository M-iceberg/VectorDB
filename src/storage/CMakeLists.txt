add_library(vortex_storage STATIC
    wal.cpp
    vector_file.cpp
    graph_serializer.cpp
    metadata_index.cpp
)
target_include_directories(vortex_storage PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(vortex_storage PUBLIC vortex_core)
