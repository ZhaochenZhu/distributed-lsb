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
set(CMAKE_CUDA_ARCHITECTURES 70 80)

enable_language(CUDA)

# Find all radix_sort_*.cu files
file(GLOB RADIX_SORT_SOURCES "radix_sort_*.cu")

# Link against cuRAND
find_package(CUDAToolkit REQUIRED)

foreach(src ${RADIX_SORT_SOURCES})
    # Get the filename without extension
    get_filename_component(target_name ${src} NAME_WE)

    # Create executable
    add_executable(${target_name} ${src})

    # Set CUDA properties
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    # Link necessary CUDA libraries
    target_link_libraries(${target_name} PRIVATE CUDA::cudart CUDA::curand)

    # Install rule
    install(TARGETS ${target_name} RUNTIME DESTINATION bin)
endforeach()
