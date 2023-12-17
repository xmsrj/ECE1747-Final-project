cmake_minimum_required(VERSION 3.0)
project(pr)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(serial serial/pagerank_serial.cpp)
add_executable(omp openMP/pagerank_openmp.cpp)
add_executable(mpi MPI/pagerank_mpi.cpp)
target_link_libraries(mpi PUBLIC MPI::MPI_CXX MPI::MPI_C)