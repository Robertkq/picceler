add_executable(tests INTERFACE)

target_link_libraries(tests PUBLIC
    PiccelerFrontend
    GTest::gtest_main
)

set_target_properties(tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)