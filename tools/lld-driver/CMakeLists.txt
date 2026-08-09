add_executable(lld-driver
    main.cpp
)

target_link_libraries(lld-driver PRIVATE
    lldELF
    lldCommon
    LLVMSupport
)

set_target_properties(lld-driver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)