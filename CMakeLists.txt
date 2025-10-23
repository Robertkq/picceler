cmake_minimum_required(VERSION 3.20)

project(picceler)

set(CMAKE_CXX_STANDARD 20)

option(ENABLE_TESTS "Enable testing" ON)
option(ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

find_package(CLI11 REQUIRED)
find_package(spdlog REQUIRED)

if (ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy)
    if(CLANG_TIDY_EXE)
        message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-checks=readability-identifier-naming")
    else()
        message(WARNING "clang-tidy option enabled but clang-tidy executable not found!")
    endif()
endif()

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

set(COMPILER_NAME "picceler")

if (ENABLE_TESTS)
  add_subdirectory(tests)
endif()

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
    CLI11::CLI11
    spdlog::spdlog
    MLIRIR
    MLIRFuncDialect
    MLIRArithDialect
    MLIRParser
    MLIRSupport
    MLIRTransforms
    MLIRPass
    MLIRDialect
)
