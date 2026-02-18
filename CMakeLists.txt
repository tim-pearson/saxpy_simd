cmake_minimum_required(VERSION 3.16)
project(KokkosSaxpySIMD)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-tree-vectorize")
##set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fno-tree-vectorize")


find_package(Kokkos REQUIRED)

include_directories(include) 


file(GLOB_RECURSE sources src/*.cc)


add_executable(SAXPY_SIMD ${sources})
target_link_libraries(SAXPY_SIMD
    PRIVATE
    Kokkos::kokkos
)

