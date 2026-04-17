add_library(vortex_tools STATIC load_sift.cpp)
target_include_directories(vortex_tools PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(vortex_tools PUBLIC cxx_std_17)
