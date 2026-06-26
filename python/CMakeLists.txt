pybind11_add_module(_vectordb bindings.cpp)

target_include_directories(_vectordb PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${Python3_NumPy_INCLUDE_DIRS}
)

target_link_libraries(_vectordb PRIVATE
    vortex_core vortex_storage vortex_server
)

# Place the .so directly inside python/vectordb/ so `import vectordb` works
# from the python/ directory without any install step.
set_target_properties(_vectordb PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/vectordb
)
