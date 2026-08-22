file(GLOB E2E_CASES "${CMAKE_CURRENT_SOURCE_DIR}/*.pic")

add_custom_target(copy-e2e-files ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/e2e"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${E2E_CASES} "${CMAKE_BINARY_DIR}/e2e"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/lit.local.cfg"
        "${CMAKE_BINARY_DIR}/e2e/lit.local.cfg"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/picceler-wrapper.sh"
        "${CMAKE_BINARY_DIR}/e2e/picceler-wrapper.sh"
    COMMAND chmod +x
        "${CMAKE_BINARY_DIR}/e2e/picceler-wrapper.sh"
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "${CMAKE_BINARY_DIR}/e2e/img"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/img/cat.png"
        "${CMAKE_SOURCE_DIR}/img/blue.png"
        "${CMAKE_SOURCE_DIR}/img/white.png"
        "${CMAKE_BINARY_DIR}/e2e/img"
    COMMENT "Copying e2e test files to build directory..."
    VERBATIM
)
