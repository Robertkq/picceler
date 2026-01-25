

add_executable(unittests
    src/main.cpp
    src/lexer_tests.cpp
    src/parser_tests.cpp
)

find_package(GTest REQUIRED)

target_link_libraries(unittests PUBLIC
    PiccelerFrontend
    GTest::GTest
)

set_target_properties(unittests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_custom_command(TARGET unittests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/data"
        "${CMAKE_BINARY_DIR}/data"
    COMMENT "Copying data folder to build directory..."
)

