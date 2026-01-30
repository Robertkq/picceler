
add_executable(picceler-mlir-lsp-server 
    main.cpp
)

target_link_libraries(picceler-mlir-lsp-server PRIVATE
    PiccelerIR
    PiccelerTransforms
    PiccelerGlobalDeps
    MLIRLspServerLib
    MLIRFuncToLLVM
    MLIRLLVMCommonConversion
    MLIRIR
    MLIRSupport
    LLVMSupport
    MLIRMlirOptMain
    MLIROptLib
    MLIRArithDialect
    MLIRAffineDialect
    MLIRMemRefDialect
    MLIRFuncDialect
    MLIRControlFlowDialect
    MLIRSCFDialect
)

set_target_properties(picceler-mlir-lsp-server PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)