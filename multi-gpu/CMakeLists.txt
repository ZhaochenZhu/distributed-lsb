cmake_minimum_required(VERSION 3.11)
project(radix_sort_multi_gpu)
enable_language(CUDA)
# C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Find MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH} src)

if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()


# Add the main executable
add_executable(radix_sort
  src/main.cpp
  src/kernel_driver.h
  src/radix_kernels.cu
)

target_compile_features(radix_sort PRIVATE cxx_std_14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")
set_target_properties(radix_sort PROPERTIES CUDA_ARCHITECTURES "80")
target_include_directories(radix_sort PRIVATE ${NCCL_ROOT}/include src)
target_link_libraries(radix_sort
  PRIVATE
    MPI::MPI_CXX
)

configure_file(job-run job-run COPYONLY)
