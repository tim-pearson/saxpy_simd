cmake_minimum_required(VERSION 3.16)
project(KokkosSaxpySIMD LANGUAGES CXX)

find_package(Kokkos REQUIRED)

include_directories(include)

file(GLOB_RECURSE sources src/*.cc)

add_executable(saxpy_optimized ${sources})
target_compile_options(saxpy_optimized
    PRIVATE
        -O2
        -march=native
)

target_compile_definitions(saxpy_optimized PRIVATE BUILD_NAME="vectorizeO2")
target_link_libraries(saxpy_optimized
    PRIVATE
        Kokkos::kokkos
)


# Scalar build
add_executable(saxpy_scalar ${sources})
target_compile_options(saxpy_scalar
    PRIVATE
        -fno-tree-vectorize
)
target_compile_definitions(saxpy_scalar PRIVATE BUILD_NAME="no-vectorizeO2")
target_link_libraries(saxpy_scalar
    PRIVATE
        Kokkos::kokkos
)
