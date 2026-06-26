# engine.cpp added Day 15 (checkpoint + recovery)
# grpc_server.cpp added Optional / Phase 1.5

add_library(vortex_server STATIC placeholder.cpp engine.cpp)
target_include_directories(vortex_server PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(vortex_server PUBLIC vortex_core vortex_storage)
