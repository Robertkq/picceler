find_package(OpenCV REQUIRED)

add_executable(reference_bench
    reference_bench.cpp
)

target_link_libraries(reference_bench PRIVATE
    ${OpenCV_LIBS}
)

set_target_properties(reference_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
