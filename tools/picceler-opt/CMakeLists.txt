
add_executable(picceler-opt 
    main.cpp
)

target_link_libraries(picceler-opt PRIVATE
    PiccelerIR
    PiccelerTransforms
    PiccelerGlobalDeps
    MLIRMlirOptMain
    MLIROptLib
    MLIRArithDialect
    MLIRAffineDialect
    MLIRMemRefDialect
    MLIRFuncDialect
    MLIRControlFlowDialect
    MLIRSCFDialect
)

set_target_properties(picceler-opt PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)