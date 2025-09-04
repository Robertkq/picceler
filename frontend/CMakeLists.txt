
file(GLOB FRONTEND_SRC src/*.cpp)
list(REMOVE_ITEM FRONTEND_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

file(GLOB FRONTEND_HDR include/*.h)

add_library(frontend_lib ${FRONTEND_SRC} ${FRONTEND_HDR})

target_include_directories(frontend_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

