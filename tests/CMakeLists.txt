
file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp")
file(GLOB TEST_SOURCES "src/*.cpp")

list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../src/main.cpp")

add_executable(tests ${TEST_SOURCES} ${SOURCES})

find_package(CLI11 REQUIRED)
find_package(spdlog REQUIRED)
find_package(GTest REQUIRED)
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

target_include_directories(tests PRIVATE
 ${CMAKE_CURRENT_SOURCE_DIR}/../include
 ${CMAKE_CURRENT_BINARY_DIR}/..
 ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tests PRIVATE
    CLI11::CLI11
    spdlog::spdlog
    GTest::GTest
    MLIRIR
    MLIRFuncDialect
    MLIRArithDialect
    MLIRParser
    MLIRSupport
    MLIRTransforms
    MLIRPass
    MLIRDialect
    MLIROptLib
    MLIRAnalysis
    LLVMCore
    LLVMSupport
    MLIRLLVMCommonConversion
    MLIRConvertToLLVMPass
    MLIRFuncToLLVM
    MLIRTargetLLVMIRExport
    MLIRLLVMToLLVMIRTranslation
    MLIRBuiltinToLLVMIRTranslation
    LLVMX86CodeGen
    LLVMX86Desc
    LLVMX86Info
    LLVMX86TargetMCA
    LLVMX86AsmParser
    LLVMX86Disassembler
)

add_custom_command(
    TARGET tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/data
            $<TARGET_FILE_DIR:tests>/data
    COMMENT "Copying test data folder to build directory"
)