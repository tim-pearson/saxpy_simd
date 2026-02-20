cmake_minimum_required(VERSION 3.16)
project(KokkosSaxpySIMD LANGUAGES CXX)

set(COMMON_FLAGS "-O3" "-march=native")



find_package(Kokkos REQUIRED)

include_directories(include)

file(GLOB_RECURSE sources src/*.cc)


# build with auto-vectorization enabled
add_executable(saxpy_vectorized ${sources})
target_compile_options(saxpy_vectorized PRIVATE ${COMMON_FLAGS} -ftree-vectorize)

target_compile_definitions(saxpy_vectorized PRIVATE BUILD_NAME="vectorize")
target_link_libraries(saxpy_vectorized Kokkos::kokkos)

# build with auto-vectorization disabled
add_executable(saxpy_no_vectorize ${sources})
target_compile_options(saxpy_no_vectorize PRIVATE ${COMMON_FLAGS} -fno-tree-vectorize)
target_compile_definitions(saxpy_no_vectorize PRIVATE BUILD_NAME="no-vectorize")
target_link_libraries(saxpy_no_vectorize Kokkos::kokkos)
