// File: radix_sort_multi.cu

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <cassert>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

#define RADIX 8
#define N_BUCKETS (1 << RADIX)
#define N_DIGITS (64 / RADIX)
#define TPB 256 // threads per block

// Kernel: per-block local histogram (Grid-stride loop)
__global__ void computeLocalCounts(
    const uint64_t *keys, int *counts, int n, int digit)
{
    extern __shared__ int s_counts[];
    int lane = threadIdx.x;
    // zero-init shared histogram
    for (int b = lane; b < N_BUCKETS; b += blockDim.x)
        s_counts[b] = 0;
    __syncthreads();

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t idx = tid; idx < (size_t)n; idx += stride)
    {
        uint64_t key = keys[idx];
        int bucket = (key >> (digit * RADIX)) & (N_BUCKETS - 1);
        atomicAdd(&s_counts[bucket], 1);
    }
    __syncthreads();

    // write per-block histogram out
    for (int b = lane; b < N_BUCKETS; b += blockDim.x)
    {
        counts[blockIdx.x * N_BUCKETS + b] = s_counts[b];
    }
}

// Kernel: per-block stable scatter using precomputed per-block offsets
__global__ void scatterStablePerBlock(
    const uint64_t *in_keys, const uint64_t *in_vals,
    uint64_t *out_keys, uint64_t *out_vals,
    const int *perBlockOffsets, int n, int digit)
{
    int lane = threadIdx.x;
    int bidx = blockIdx.x;
    int start = bidx * blockDim.x;
    int end = min(n, start + blockDim.x);

    // thread-0 does the stable write for its block
    if (lane == 0)
    {
        int localCnt[N_BUCKETS] = {0};
        for (int idx = start; idx < end; ++idx)
        {
            uint64_t key = in_keys[idx];
            uint64_t val = in_vals[idx];
            int bucket = (key >> (digit * RADIX)) & (N_BUCKETS - 1);

            int base = perBlockOffsets[bidx * N_BUCKETS + bucket];
            int pos = base + localCnt[bucket]++;

            out_keys[pos] = key;
            out_vals[pos] = val;
        }
    }
}

// Kernel: init cuRAND states
__global__ void initRNG(curandState *states, unsigned long seed, int n)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride)
    {
        curand_init(seed, i, 0, &states[i]);
    }
}

// Kernel: generate random keys and sequential vals
__global__ void generateRandom(
    uint64_t *d_keys, uint64_t *d_vals,
    curandState *states, int n)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride)
    {
        // combine two 32-bit draws into 64-bit
        uint64_t r = ((uint64_t)curand(&states[i]) << 32) |
                     (uint64_t)curand(&states[i]);
        d_keys[i] = r;
        d_vals[i] = i;
    }
}

// Host: perform one digit of stable bucket sort on GPU with per-block offsets
template <typename T>
void gpuShufflePerBlock(
    const T *d_in_keys, const T *d_in_vals,
    T *d_out_keys, T *d_out_vals,
    int *d_counts, int *d_perBlockOffsets,
    int n, int digit)
{
    int blocks = (n + TPB - 1) / TPB;

    // 1) compute per-block histograms
    computeLocalCounts<<<blocks, TPB, N_BUCKETS * sizeof(int)>>>(
        d_in_keys, d_counts, n, digit);

    // 2) copy back per-block histograms to host
    std::vector<int> h_counts(blocks * N_BUCKETS);
    cudaMemcpy(h_counts.data(), d_counts,
               sizeof(int) * blocks * N_BUCKETS,
               cudaMemcpyDeviceToHost);

    // 3) build global prefix sum + per-block offsets
    std::vector<int> h_global(N_BUCKETS, 0);
    for (int b = 0; b < blocks; ++b)
        for (int i = 0; i < N_BUCKETS; ++i)
            h_global[i] += h_counts[b * N_BUCKETS + i];
    std::vector<int> h_starts(N_BUCKETS);
    int sum = 0;
    for (int i = 0; i < N_BUCKETS; ++i)
    {
        h_starts[i] = sum;
        sum += h_global[i];
    }
    std::vector<int> h_perBlockOffsets(blocks * N_BUCKETS);
    for (int i = 0; i < N_BUCKETS; ++i)
    {
        int running = h_starts[i];
        for (int b = 0; b < blocks; ++b)
        {
            h_perBlockOffsets[b * N_BUCKETS + i] = running;
            running += h_counts[b * N_BUCKETS + i];
        }
    }
    cudaMemcpy(d_perBlockOffsets, h_perBlockOffsets.data(),
               sizeof(int) * blocks * N_BUCKETS,
               cudaMemcpyHostToDevice);

    // 4) stable scatter in each block
    scatterStablePerBlock<<<blocks, TPB>>>(
        d_in_keys, d_in_vals,
        d_out_keys, d_out_vals,
        d_perBlockOffsets, n, digit);
    cudaDeviceSynchronize();
}

int main(int argc, char **argv)
{
    // Query GPU devices
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Number of CUDA devices: " << deviceCount << "\n";
    if (deviceCount > 0)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Using GPU [0]: " << prop.name << "\n";
    }

    // Problem size
    int64_t n = 10'000'000;
    if (argc > 1)
        n = atoll(argv[1]);
    std::cout << "Problem size: " << n << "\n";

    // Device allocation
    uint64_t *d_in_keys, *d_out_keys;
    uint64_t *d_in_vals, *d_out_vals;
    int *d_counts, *d_perBlockOffsets;
    curandState *d_states;

    cudaMalloc(&d_in_keys, sizeof(uint64_t) * n);
    cudaMalloc(&d_out_keys, sizeof(uint64_t) * n);
    cudaMalloc(&d_in_vals, sizeof(uint64_t) * n);
    cudaMalloc(&d_out_vals, sizeof(uint64_t) * n);
    cudaMalloc(&d_counts, sizeof(int) * ((n + TPB - 1) / TPB) * N_BUCKETS);
    cudaMalloc(&d_perBlockOffsets, sizeof(int) * ((n + TPB - 1) / TPB) * N_BUCKETS);
    cudaMalloc(&d_states, sizeof(curandState) * n);

    int blocks = (n + TPB - 1) / TPB;

    // Generate random values on GPU
    initRNG<<<blocks, TPB>>>(d_states, 42ULL, n);
    cudaDeviceSynchronize();
    std::cout << "Generating random values on GPU\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    generateRandom<<<blocks, TPB>>>(d_in_keys, d_in_vals, d_states, n);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gen_sec = t1 - t0;
    std::cout << "Generated random values in "
              << gen_sec.count() << " s on GPU\n";

    // Sorting
    std::cout << "Sorting\n";
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int digit = 0; digit < N_DIGITS; ++digit)
    {
        gpuShufflePerBlock<uint64_t>(
            d_in_keys, d_in_vals,
            d_out_keys, d_out_vals,
            d_counts, d_perBlockOffsets,
            n, digit);
        std::swap(d_in_keys, d_out_keys);
        std::swap(d_in_vals, d_out_vals);
    }
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sort_sec = t3 - t2;
    std::cout << "Sorted " << n << " values in "
              << sort_sec.count() << " s\n";

    // Throughput
    double rate = double(n) / sort_sec.count() / 1e6;
    std::cout << "That's " << rate
              << " M elements sorted / s\n";

    // Verify
    std::vector<uint64_t> h_keys(n);
    cudaMemcpy(h_keys.data(), d_in_keys,
               sizeof(uint64_t) * n, cudaMemcpyDeviceToHost);
    // for (int64_t i = 1; i < n; ++i)
    //     assert(h_keys[i-1] <= h_keys[i]);
    thrust::device_vector<uint64_t> dv(h_keys.begin(), h_keys.end());
    bool is_sorted = thrust::is_sorted(dv.begin(), dv.end());
    assert(is_sorted);

    // Cleanup
    cudaFree(d_in_keys);
    cudaFree(d_out_keys);
    cudaFree(d_in_vals);
    cudaFree(d_out_vals);
    cudaFree(d_counts);
    cudaFree(d_perBlockOffsets);
    cudaFree(d_states);

    return 0;
}