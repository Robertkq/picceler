

add_executable(unittests
    src/main.cpp
    src/lexer_tests.cpp
    src/parser_tests.cpp
)

find_package(GTest REQUIRED)
find_package(spdlog REQUIRED)

target_link_libraries(unittests PUBLIC
    PiccelerFrontend
    GTest::GTest
    spdlog::spdlog
)

set_target_properties(unittests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

