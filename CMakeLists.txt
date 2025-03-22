cmake_minimum_required(VERSION 3.31)
project(minEdge)

set(CMAKE_CXX_STANDARD 20)

add_executable(minEdge taskparallel.cpp)
add_executable(noparallel noparallel.cpp)
add_executable(taskparallel taskparallel.cpp)
find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    target_link_libraries(minEdge PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(taskparallel PUBLIC OpenMP::OpenMP_CXX)
endif()
