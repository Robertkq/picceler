

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

add_library(PiccelerOps STATIC
    ops.cpp
    ops/common.cpp
    ops/convolution.cpp
    ops/dilate.cpp
    ops/erode.cpp
    ops/blend.cpp
    ops/diff.cpp
    ops/print.cpp
    ops/brightness.cpp
    ops/invert.cpp
    ops/sharpen.cpp
    ops/box_blur.cpp
    ops/gaussian_blur.cpp
    ops/edge_detect.cpp
    ops/emboss.cpp
    ops/rotate.cpp
    ops/crop.cpp
)

target_link_libraries(PiccelerOps PUBLIC
    PiccelerGlobalDeps
    MLIRIR
    MLIRSupport
    MLIRParser
    MLIRAffineDialect
    MLIRMemRefDialect
)

add_dependencies(PiccelerOps
    ${PICCELER_TABLEGEN_TARGETS}
)

add_library(PiccelerIR STATIC
    dialect.cpp
    types.cpp
    mlir_gen.cpp
)

target_link_libraries(PiccelerIR PUBLIC
    PiccelerGlobalDeps
    PiccelerFrontend
    PiccelerOps
    MLIRIR
    MLIRSupport
    MLIRParser
)

add_library(PiccelerTransforms STATIC
    generic_type_rewriter.cpp
    picceler_kernel_to_memref_pass.cpp
    picceler_ops_to_func_calls_pass.cpp
    picceler_to_affine_pass.cpp
    picceler_to_llvm_ir_pass.cpp
    picceler_filters_to_conv_pass.cpp
    picceler_add_profiling_pass.cpp
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
    MLIRMathDialect
    PiccelerSupport
    PiccelerOps
)

add_library(PiccelerSupport STATIC
    image_access_helper.cpp
)

target_link_libraries(PiccelerSupport PUBLIC
    PiccelerGlobalDeps
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
    MLIRAffineTransforms
    MLIROptLib
    MLIRTargetLLVMIRExport
    MLIRLLVMToLLVMIRTranslation
    MLIRBuiltinToLLVMIRTranslation
    MLIRConvertToLLVMPass
    MLIRFuncToLLVM
    MLIRMemRefToLLVM
    MLIRArithToLLVM
    MLIRMathToLLVM
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

add_custom_target(copy-img ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/img"
        "${CMAKE_BINARY_DIR}/img"
    COMMENT "Copying img/ to build directory"
    VERBATIM
)

add_dependencies(picceler copy-img)
