#cmake_minimum_required(VERSION 3.16)
#project(KokkosSaxpySIMD)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3")
#find_package(Kokkos REQUIRED)
#add_executable(SAXPY_SIMD main.cpp)
#target_link_libraries(SAXPY_SIMD Kokkos::kokkos)

cmake_minimum_required(VERSION 3.16)
project(KokkosSaxpySIMD)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(Kokkos REQUIRED)
find_package(Matplot++ REQUIRED)  # If using Matplot++


file(GLOB_RECURSE sources src/*.cc)


add_executable(SAXPY_SIMD ${sources})
target_link_libraries(SAXPY_SIMD
    PRIVATE
    Kokkos::kokkos
    Matplot++::matplot  # Link Matplot++ if used
)

