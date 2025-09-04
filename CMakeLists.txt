cmake_minimum_required(VERSION 3.2)

project(picceler)

set(CMAKE_CXX_STANDARD 20)

option(ENABLE_TESTS "Enable testing" ON)
option(ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)

find_package(CLI11 REQUIRED)
find_package(spdlog REQUIRED)

if (ENABLE_TESTS)
  # add_subdirectory(tests)
endif()

if (ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy)
    if(CLANG_TIDY_EXE)
        message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-checks=readability-identifier-naming")
    else()
        message(WARNING "clang-tidy option enabled but clang-tidy executable not found!")
    endif()
endif()

add_subdirectory(frontend)
add_subdirectory(middleend)
add_subdirectory(backend)

set(COMPILER_NAME "picceler")

add_executable(${COMPILER_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/frontend/src/main.cpp
)
target_include_directories(${COMPILER_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/frontend/include)
target_link_libraries(${COMPILER_NAME} PRIVATE frontend_lib CLI11::CLI11 spdlog::spdlog)
