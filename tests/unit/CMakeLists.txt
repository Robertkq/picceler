
file(GLOB_RECURSE LEXER_TEST_SOURCES
    src/lexer/*.cpp
)

file(GLOB_RECURSE PARSER_TEST_SOURCES
    src/parser/*.cpp
)

file(GLOB_RECURSE AST_TEST_SOURCES
    src/ast/*.cpp
)

add_executable(unittests
    src/main.cpp
    ${LEXER_TEST_SOURCES}
    ${PARSER_TEST_SOURCES}
    ${AST_TEST_SOURCES}
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

