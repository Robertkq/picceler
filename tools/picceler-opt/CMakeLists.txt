
add_executable(picceler-opt ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_include_directories(picceler-opt PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(picceler-opt PRIVATE
    MLIROptLib
    MLIRArithDialect
    MLIRAffineDialect
    MLIRMemRefDialect
    MLIRFuncDialect
    MLIRControlFlowDialect
    MLIRSCFDialect
)