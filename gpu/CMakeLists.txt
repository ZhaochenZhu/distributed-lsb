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
file(GLOB RADIX_SORT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/radix_sort_*.cu")

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

# ----------------------------------------------------------------------------
# Copy generate_timing_script.sh into the build dir and make it executable
# ----------------------------------------------------------------------------

# This custom command will copy the script from source → build and chmod +x it
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generate_timing_script.sh"
  COMMAND ${CMAKE_COMMAND} -E copy
          "${CMAKE_CURRENT_SOURCE_DIR}/generate_timing_script.sh"
          "${CMAKE_CURRENT_BINARY_DIR}/generate_timing_script.sh"
  COMMAND ${CMAKE_COMMAND} -E chmod +x
          "${CMAKE_CURRENT_BINARY_DIR}/generate_timing_script.sh"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/generate_timing_script.sh"
  COMMENT "Copying generate_timing_script.sh into build directory"
  VERBATIM
)

# Tie that command into an ALL target so it runs on 'make'
add_custom_target(copy_generate_script ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/generate_timing_script.sh"
  COMMENT "Ensuring generate_timing_script.sh is available in build/"
)

# Optionally install the helper script as well
install(
  PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/generate_timing_script.sh"
  DESTINATION bin
)
