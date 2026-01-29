

add_library(PiccelerGlobalDeps INTERFACE) 

target_include_directories(PiccelerGlobalDeps INTERFACE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/tablegen
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(PiccelerGlobalDeps INTERFACE 
    spdlog::spdlog
)

add_library(PiccelerUtils STATIC
    utils.cpp
)

target_link_libraries(PiccelerUtils PUBLIC
    PiccelerGlobalDeps
)

add_library(PiccelerFrontend STATIC
    lexer.cpp
    ast.cpp
    parser.cpp
)

target_link_libraries(PiccelerFrontend PUBLIC 
    PiccelerGlobalDeps
    PiccelerUtils
)

add_library(PiccelerIR STATIC
    dialect.cpp
    types.cpp
    ops.cpp
    mlir_gen.cpp
)

target_link_libraries(PiccelerIR PUBLIC
    PiccelerGlobalDeps 
    PiccelerFrontend
    MLIRIR
    MLIRSupport
    MLIRParser
)

add_dependencies(PiccelerIR 
    ${PICCELER_TABLEGEN_TARGETS}
)

add_library(PiccelerTransforms STATIC
    picceler_kernel_to_memref_pass.cpp
    picceler_ops_to_func_calls_pass.cpp
    picceler_to_affine_pass.cpp
    picceler_to_llvm_ir_pass.cpp
    picceler_filters_to_conv_pass.cpp
)

target_link_libraries(PiccelerTransforms PUBLIC 
    PiccelerIR
    MLIRPass
    MLIRTransforms
    MLIRAffineDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRLLVMDialect
)


add_library(PiccelerSupport STATIC
    image_access_helper.cpp 
)

add_library(PiccelerCore STATIC
    compiler.cpp
    pass_manager.cpp
)

target_link_libraries(PiccelerCore PUBLIC
    CLI11::CLI11
    PiccelerFrontend
    PiccelerIR
    PiccelerTransforms
    PiccelerSupport
    MLIROptLib
    MLIRTargetLLVMIRExport
    MLIRLLVMToLLVMIRTranslation
    MLIRBuiltinToLLVMIRTranslation
    MLIRConvertToLLVMPass
    MLIRFuncToLLVM
    MLIRMemRefToLLVM
    MLIRArithToLLVM
    MLIRSCFToControlFlow
    MLIRReconcileUnrealizedCasts
    MLIRAffineToStandard
    # hardware libs
    ${NATIVE_HW_LLVM_LIBS}
)

add_executable(picceler 
    main.cpp
)

set_target_properties(picceler PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_link_libraries(picceler
    PiccelerCore
)