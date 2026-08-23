set(_copy_lit_files_commands
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
        "${CMAKE_BINARY_DIR}/mlir"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py"
        "${CMAKE_BINARY_DIR}/lit.cfg.py"
)

if (ENABLE_E2E_TESTS)
    file(GLOB E2E_CASES "${CMAKE_CURRENT_SOURCE_DIR}/e2e/*.pic")

    list(APPEND _copy_lit_files_commands
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/e2e"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${E2E_CASES} "${CMAKE_BINARY_DIR}/e2e"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/e2e/lit.local.cfg"
            "${CMAKE_BINARY_DIR}/e2e/lit.local.cfg"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/e2e/img"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_SOURCE_DIR}/img/cat.png"
            "${CMAKE_SOURCE_DIR}/img/blue.png"
            "${CMAKE_SOURCE_DIR}/img/white.png"
            "${CMAKE_BINARY_DIR}/e2e/img"
    )
endif()

add_custom_target(copy-lit-files ALL
    ${_copy_lit_files_commands}
    COMMENT "Copying LIT test files to build directory..."
    VERBATIM
)

find_program(LIT_EXECUTABLE lit)
if (NOT LIT_EXECUTABLE)
    message(WARNING "lit not found; 'mlir' and 'e2e' test-running targets will not be available")
endif()

if (LIT_EXECUTABLE)
    add_custom_target(mlir
        COMMAND ${LIT_EXECUTABLE} -v "${CMAKE_BINARY_DIR}/mlir"
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMMENT "Running MLIR lit tests"
        USES_TERMINAL
    )
    add_dependencies(mlir picceler-opt copy-lit-files)

    if (ENABLE_E2E_TESTS)
        add_custom_target(e2e
            COMMAND ${LIT_EXECUTABLE} -v "${CMAKE_BINARY_DIR}/e2e"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            COMMENT "Running e2e lit tests"
            USES_TERMINAL
        )
        add_dependencies(e2e picceler PiccelerRuntime copy-lit-files)
    endif()
endif()
