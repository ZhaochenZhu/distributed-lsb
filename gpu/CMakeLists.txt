cmake_minimum_required(VERSION 3.11)
project(radix_sort LANGUAGES CXX CUDA)

# Compiler standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED YES)

# Build type default to Release for performance
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Specify CUDA architectures (adjust this to your GPU)
# Examples: 70 for Volta, 80 for Ampere
set(CMAKE_CUDA_ARCHITECTURES 70 80)

enable_language(CUDA)

# Add the executables
add_executable(radix_sort_multi radix_sort_multi.cu)
add_executable(radix_sort_single radix_sort_single.cu)

# Allow separable compilation
set_target_properties(radix_sort_multi PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(radix_sort_single PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link against cuRAND for random number generation
find_package(CUDAToolkit REQUIRED)
if (CUDAToolkit_FOUND)
  target_link_libraries(radix_sort_multi PRIVATE CUDA::cudart CUDA::curand)
  target_link_libraries(radix_sort_single PRIVATE CUDA::cudart CUDA::curand)
endif()

# Installation rules (optional)
install(TARGETS radix_sort_multi radix_sort_single
    RUNTIME DESTINATION bin
)
