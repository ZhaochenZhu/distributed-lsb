# Distributed LSD Radix Sort

This repository contains our multi-GPU implementation of least-significant-digit
radix sort using CUDA kernels for local work and CUDA-aware MPI for
inter-GPU coordination.

## Build & Run (Perlmutter GPU nodes)

```bash
# 1. configure an out-of-source build
mkdir build
cd build

# 2. generate Makefiles in Release mode
cmake -DCMAKE_BUILD_TYPE=Release ..

# 3. compile (parallel make)
make -j

# 4. obtain an interactive GPU allocation
salloc -A mp309 -N 1 -C gpu -q interactive \
       -t 00:10:00 --gpus-per-node=4

# 5. run radix sort on 4 GPUs
#    (replace 1000000 with any problem size n)
srun --ntasks=4 --gpus-per-task=1 ./radix_sort 1000000
