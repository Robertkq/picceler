add_custom_target(copy-lit-files ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
        "${CMAKE_BINARY_DIR}/mlir"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py"
        "${CMAKE_BINARY_DIR}/lit.cfg.py"
    COMMENT "Copying LIT test files to build directory..."
    VERBATIM
)