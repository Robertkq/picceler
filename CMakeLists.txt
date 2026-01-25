cmake_minimum_required(VERSION 3.20)

project(picceler)

set(CMAKE_CXX_STANDARD 23)

option(ENABLE_TESTS "Enable testing" ON)
option(ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
option(ENABLE_DOCS "Enable documentation generation" OFF)

if(NOT DEFINED CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Generate compile_commands.json" FORCE)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

find_package(CLI11 REQUIRED)
find_package(spdlog REQUIRED)

if (ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy)
    if(CLANG_TIDY_EXE)
        message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-extra-arg=-std=c++${CMAKE_CXX_STANDARD}")
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

llvm_map_components_to_libnames(NATIVE_HW_LLVM_LIBS 
    X86 
    Target 
    Core 
    Support 
    AsmPrinter
)

message(STATUS "Native hardware LLVM libs: ${NATIVE_HW_LLVM_LIBS}")

add_subdirectory(lib)

add_subdirectory(tablegen)

add_subdirectory(src)

add_subdirectory(tools)

if (ENABLE_TESTS)
  add_subdirectory(tests)
endif()




