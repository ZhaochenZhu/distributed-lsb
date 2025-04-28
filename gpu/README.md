# README

## Project Description

This project implements two versions of Radix Sort on GPU using CUDA:

- **radix_sort_single**: Each thread processes a single key through all steps (extracting the last digit, histogramming, prefix sum, scattering, etc.). This version cannot scale to large input sizes and is only suitable for sorting a small number of keys.

- **radix_sort_multi**: Designed for better scalability, this version partitions work across threads and blocks more efficiently, allowing the sorting of much larger input sizes.

---

## Build Instructions

1. Load the required module:
   ```bash
   module load cmake
   ```

2. Create a build directory and navigate into it:
   ```bash
   mkdir build
   cd build
   ```

3. Configure the project with CMake:
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Release ..
   ```

4. Build the project:
   ```bash
   make
   ```

---

## Running Instructions

1. Request an interactive GPU node:
   ```bash
   salloc -A mp309 -N 1 -C gpu -q interactive -t 00:30:00
   ```

2. Once inside the node, run the executables. For example:

   To run the scalable multi-threaded version:
   ```bash
   ./radix_sort_multi 1000000
   ```

   To run the single-threaded-per-key version (for small inputs only):
   ```bash
   ./radix_sort_single 1000
   ```

---

## Notes

- Ensure you are on a GPU node before executing the program.
- Adjust the input size argument as needed to test performance with different data sizes.
- **radix_sort_single** is intended for small tests only due to its lack of scalability.
- **radix_sort_multi** can handle much larger datasets effectively.
