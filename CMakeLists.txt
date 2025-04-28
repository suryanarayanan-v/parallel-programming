cmake_minimum_required(VERSION 3.31)
project(minEdge)

set(CMAKE_CXX_STANDARD 20)

add_executable(minEdge taskparallel.cpp)
add_executable(noparallel noparallel.cpp)
add_executable(taskparallel taskparallel.cpp)
add_executable(dataparallel dataparallelsim.cpp)
add_executable(mpibnb mpibnb.cpp)
add_executable(impmpi hybrid.cpp)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

if(OpenMP_CXX_FOUND)
    target_link_libraries(minEdge PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(taskparallel PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(dataparallel PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(impmpi PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(mpibnb PUBLIC MPI::MPI_CXX)
target_link_libraries(impmpi PUBLIC MPI::MPI_CXX)

