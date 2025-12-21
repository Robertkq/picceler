cmake_minimum_required(VERSION 3.20)

project(picceler_runtime)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)

file(GLOB SOURCES "src/*.cpp")

add_library(picceler_runtime STATIC
    ${SOURCES}
)

target_include_directories(picceler_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(picceler_runtime
    ${OpenCV_LIBS}
)
