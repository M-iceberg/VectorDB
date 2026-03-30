add_subdirectory(core)
add_subdirectory(storage)
add_subdirectory(server)

add_executable(vectordb main.cpp)
target_link_libraries(vectordb PRIVATE vortex_server)
