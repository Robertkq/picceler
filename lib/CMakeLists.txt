cmake_minimum_required(VERSION 3.20)

set(RUNTIME_LIB_NAME "PiccelerRuntime")

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)

file(GLOB SOURCES "src/*.cpp")

add_library(${RUNTIME_LIB_NAME} STATIC
    ${SOURCES}
)

target_include_directories(${RUNTIME_LIB_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${RUNTIME_LIB_NAME}
    ${OpenCV_LIBS}
)
