cmake_minimum_required(VERSION 3.20)

project(picceler_runtime)

set(CMAKE_CXX_STANDARD 20)

add_library(picceler_runtime STATIC
    src/runtime.cpp
)

target_include_directories(picceler_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
