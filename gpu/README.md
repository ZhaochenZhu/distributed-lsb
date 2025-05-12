# README

## Project Description

This project implements multiple versions of Radix Sort on GPU using CUDA:

- **radix_sort_single**  
  Each thread processes a single key through all steps (extracting the last digit, histogramming, prefix sum, scattering, etc.). This version cannot scale to large input sizes and is only suitable for sorting a small number of keys.

- **radix_sort_base**  
  Designed for better scalability, this version partitions work across threads and blocks more efficiently, allowing the sorting of much larger input sizes.

- **radix_sort_ppsum**  
  Builds on **radix_sort_base** and moves the copyback + prefix‑sum + offset calculations entirely onto the GPU.

- **radix_sort_pscatter**  
  Further optimizes **radix_sort_ppsum** by parallelizing the scatter step across threads within each block.

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

## 5/11 Update

- **Renamed** `radix_sort_multi` → `radix_sort_base`.
- After pulling the latest changes, **re-run CMake** in your build directory:
  ```bash
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make
  ```
- This will copy the helper script `generate_timing_script.sh` into `build/`. You can then invoke:
  ```bash
  ./generate_timing_script.sh <suffix>
  ./run_timing_<suffix>.sh
  ```
  to produce and run a full timing sweep.

---

## Running Instructions

1. Request an interactive GPU node:
   ```bash
   salloc -A mp309 -N 1 -C gpu -q interactive -t 00:30:00
   ```
2. Once inside the node, run the executables. Examples:
   ```bash
   # Single-threaded-per-key (tiny n only)
   ./radix_sort_single 100

   # Baseline multi-threaded
   ./radix_sort_base 1000000

   # GPU prefix-sum optimization
   ./radix_sort_ppsum 1000000

   # Parallel-scatter + GPU prefix-sum
   ./radix_sort_pscatter 1000000
   ```
3. **Benchmarking**  
   After building, you’ll have `generate_timing_script.sh` in `build/`. To benchmark any variant:
   ```bash
   # For the ppsum variant:
   ./generate_timing_script.sh ppsum
   ./run_timing_ppsum.sh

   # For the pscatter variant:
   ./generate_timing_script.sh pscatter
   ./run_timing_pscatter.sh
   ```
   Each `run_timing_<suffix>.sh` will sweep _n_ = 10³ … 10⁸, record timings in `timing_results_<suffix>.txt`, and print progress.

---

## Performance Comparison

| Problem Size (n)    | base (s)   | ppsum (s)   | pscatter (s)   |
|---------------------|------------|-------------|----------------|
| 10³  (1,000)        | 0.000774   | 0.0010171   | 0.000567535    |
| 10⁴  (10,000)       | 0.00109859 | 0.00107331  | 0.000612592    |
| 10⁵  (100,000)      | 0.00561591 | 0.00174365  | 0.00094385     |
| 10⁶  (1,000,000)    | 0.0660885  | 0.00748684  | 0.00416942     |
| 10⁷  (10,000,000)   | 0.709487   | 0.0782407   | 0.0561626      |
| 10⁸  (100,000,000)  | 9.32418    | 0.700013    | 0.521434       |
