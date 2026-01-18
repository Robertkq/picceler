cmake_minimum_required(VERSION 3.20)

project(picceler)

set(CMAKE_CXX_STANDARD 23)

option(ENABLE_TESTS "Enable testing" ON)
option(ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
option(ENABLE_DOCS "Enable documentation generation" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

find_package(CLI11 REQUIRED)
find_package(spdlog REQUIRED)

if (ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy clang-tidy-18)
    if(CLANG_TIDY_EXE)
        message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
        # Pass the C++ standard explicitly so clang-tidy parses system headers consistently.
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-extra-arg=-std=c++20")
    else()
        message(WARNING "clang-tidy option enabled but clang-tidy executable not found!")
    endif()
endif()

if (ENABLE_DOCS)
    find_package(Doxygen REQUIRED)
    if(DOXYGEN_FOUND)
        message(STATUS "Doxygen found: ${DOXYGEN_EXECUTABLE}")
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
        add_custom_target(doc_doxygen ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM)
    else()
        message(WARNING "Doxygen option enabled but Doxygen executable not found!")
    endif()
endif()

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

set(COMPILER_NAME "picceler")

# Add the runtime library subdirectory
add_subdirectory(lib)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
message(STATUS "LLVM includes: ${LLVM_INCLUDE_DIRS}")
message(STATUS "MLIR includes: ${MLIR_INCLUDE_DIRS}")
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

message(STATUS "LLVM Definitions: ${LLVM_DEFINITIONS}")
message(STATUS "LLVM Libraries: ${LLVM_LIBS}")
message(STATUS "LLVM Binaries: ${LLVM_TOOLS_BINARY_DIR}")
message(STATUS "LLVM Include Dirs: ${LLVM_INCLUDE_DIRS}")
message(STATUS "LLVM CMake Dir: ${LLVM_DIR}")
message(STATUS "MLIR CMake Dir: ${MLIR_CMAKE_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(AddLLVM)
include(TableGen)
include(AddMLIR)

set(PICCELER_TD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tablegen)

# Dialect
set(LLVM_TARGET_DEFINITIONS ${PICCELER_TD_DIR}/dialect.td)
mlir_tablegen(piccelerDialect.h.inc -gen-dialect-decls -dialect=picceler -I${PICCELER_TD_DIR})
mlir_tablegen(piccelerDialect.cpp.inc -gen-dialect-defs -dialect=picceler -I${PICCELER_TD_DIR})
add_public_tablegen_target(PiccelerDialectIncGen)

# Types
set(LLVM_TARGET_DEFINITIONS ${PICCELER_TD_DIR}/types.td)
mlir_tablegen(piccelerTypes.h.inc -gen-typedef-decls -dialect=picceler -I${PICCELER_TD_DIR})
mlir_tablegen(piccelerTypes.cpp.inc -gen-typedef-defs -dialect=picceler -I${PICCELER_TD_DIR})
add_public_tablegen_target(PiccelerTypesIncGen)

# Ops
set(LLVM_TARGET_DEFINITIONS ${PICCELER_TD_DIR}/ops.td)
mlir_tablegen(piccelerOps.h.inc -gen-op-decls -dialect=picceler -I${PICCELER_TD_DIR})
mlir_tablegen(piccelerOps.cpp.inc -gen-op-defs -dialect=picceler -I${PICCELER_TD_DIR})
add_public_tablegen_target(PiccelerOpsIncGen)

# Define common libraries to link against
set(PICCELER_LINK_LIBS
    CLI11::CLI11
    spdlog::spdlog
    MLIRIR
    MLIRFuncDialect
    MLIRArithDialect
    MLIRAffineDialect
    MLIRSCFDialect
    MLIRReconcileUnrealizedCasts
    MLIRSCFToControlFlow
    MLIRAffineToStandard
    MLIRParser
    MLIRSupport
    MLIRTransforms
    MLIRPass
    MLIRDialect
    MLIROptLib
    MLIRAnalysis
    MLIRLLVMCommonConversion
    MLIRConvertToLLVMPass
    MLIRMemRefToLLVM
    MLIRFuncToLLVM
    MLIRTargetLLVMIRExport
    MLIRLLVMToLLVMIRTranslation
    MLIRBuiltinToLLVMIRTranslation
    LLVMSelectionDAG
    LLVMX86CodeGen
    LLVMX86Desc
    LLVMX86Info
    LLVMX86TargetMCA
    LLVMX86AsmParser
    LLVMX86Disassembler
    LLVMTarget
    LLVMTargetParser
    LLVMMC
    LLVMCore
    LLVMSupport
)

if (ENABLE_TESTS)
  add_subdirectory(tests)
endif()

add_executable(${COMPILER_NAME} 
    ${SOURCES}
)

add_dependencies(${COMPILER_NAME}
  PiccelerDialectIncGen
  PiccelerTypesIncGen
  PiccelerOpsIncGen
)
add_dependencies(tests
  PiccelerDialectIncGen
  PiccelerTypesIncGen
  PiccelerOpsIncGen
)

target_include_directories(${COMPILER_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_link_libraries(${COMPILER_NAME} PRIVATE
    picceler_runtime
    ${PICCELER_LINK_LIBS}
)
