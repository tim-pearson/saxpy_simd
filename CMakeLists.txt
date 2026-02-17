cmake_minimum_required(VERSION 3.16)
project(KokkosSaxpySIMD)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O2")
find_package(Kokkos REQUIRED)
add_executable(SAXPY_SIMD main.cpp)
target_link_libraries(SAXPY_SIMD Kokkos::kokkos)

