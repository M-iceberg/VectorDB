# wal.cpp           added Day 11–12
# vector_file.cpp   added Day 13
# graph_serializer.cpp added Day 14
# metadata_index.cpp added Day 16

add_library(vortex_storage STATIC placeholder.cpp wal.cpp vector_file.cpp)
target_include_directories(vortex_storage PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(vortex_storage PUBLIC vortex_core)
