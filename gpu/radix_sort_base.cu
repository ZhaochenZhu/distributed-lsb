// File: radix_sort_base.cu

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

struct ShuffleTiming {
    double histogram_ms = 0;
    double global_hist_ms = 0;
    double prefixsum_ms = 0;
    double offsets_ms = 0;
    double scatter_ms = 0;
};

__global__ void computeLocalCounts(const uint64_t *keys, int *counts, int n, int digit) {
    extern __shared__ int s_counts[];
    int lane = threadIdx.x;
    for (int b = lane; b < N_BUCKETS; b += blockDim.x)
        s_counts[b] = 0;
    __syncthreads();

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t idx = tid; idx < (size_t)n; idx += stride) {
        uint64_t key = keys[idx];
        int bucket = (key >> (digit * RADIX)) & (N_BUCKETS - 1);
        atomicAdd(&s_counts[bucket], 1);
    }
    __syncthreads();

    for (int b = lane; b < N_BUCKETS; b += blockDim.x)
        counts[blockIdx.x * N_BUCKETS + b] = s_counts[b];
}

__global__ void scatterStablePerBlock(const uint64_t *in_keys, const uint64_t *in_vals, uint64_t *out_keys, uint64_t *out_vals, const int *perBlockOffsets, int n, int digit) {
    int lane = threadIdx.x;
    int bidx = blockIdx.x;
    int start = bidx * blockDim.x;
    int end = min(n, start + blockDim.x);

    if (lane == 0) {
        int localCnt[N_BUCKETS] = {0};
        for (int idx = start; idx < end; ++idx) {
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

__global__ void initRNG(curandState *states, unsigned long seed, int n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride)
        curand_init(seed, i, 0, &states[i]);
}

__global__ void generateRandom(uint64_t *d_keys, uint64_t *d_vals, curandState *states, int n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        uint64_t r = ((uint64_t)curand(&states[i]) << 32) | (uint64_t)curand(&states[i]);
        d_keys[i] = r;
        d_vals[i] = i;
    }
}

template <typename T>
void gpuShufflePerBlock(const T *d_in_keys, const T *d_in_vals, T *d_out_keys, T *d_out_vals, int *d_counts, int *d_perBlockOffsets, int n, int digit, ShuffleTiming &timing) {
    int blocks = (n + TPB - 1) / TPB;
    using clk = std::chrono::high_resolution_clock;

    auto t_start = clk::now();
    // Compute local counts
    computeLocalCounts<<<blocks, TPB, N_BUCKETS * sizeof(int)>>>(d_in_keys, d_counts, n, digit);
    cudaDeviceSynchronize();
    auto t_histogram = clk::now();

    std::vector<int> h_counts(blocks * N_BUCKETS);
    // Copy back the counts to host
    cudaMemcpy(h_counts.data(), d_counts, sizeof(int) * blocks * N_BUCKETS, cudaMemcpyDeviceToHost);
 
    // Compute global counts in cpu
    std::vector<int> h_global(N_BUCKETS, 0);
    for (int b = 0; b < blocks; ++b)
        for (int i = 0; i < N_BUCKETS; ++i)
            h_global[i] += h_counts[b * N_BUCKETS + i];
    auto t_global_hist = clk::now();

    std::vector<int> h_starts(N_BUCKETS);
    int sum = 0;
    for (int i = 0; i < N_BUCKETS; ++i) {
        h_starts[i] = sum;
        sum += h_global[i];
    }
    auto t_prefixsum = clk::now();

    std::vector<int> h_perBlockOffsets(blocks * N_BUCKETS);
    for (int i = 0; i < N_BUCKETS; ++i) {
        int running = h_starts[i];
        for (int b = 0; b < blocks; ++b) {
            h_perBlockOffsets[b * N_BUCKETS + i] = running;
            running += h_counts[b * N_BUCKETS + i];
        }
    }
    cudaMemcpy(d_perBlockOffsets, h_perBlockOffsets.data(), sizeof(int) * blocks * N_BUCKETS, cudaMemcpyHostToDevice);
    auto t_offsets = clk::now();

    scatterStablePerBlock<<<blocks, TPB>>>(d_in_keys, d_in_vals, d_out_keys, d_out_vals, d_perBlockOffsets, n, digit);
    cudaDeviceSynchronize();
    auto t_scatter = clk::now();

    timing.histogram_ms += std::chrono::duration<double, std::milli>(t_histogram - t_start).count();
    timing.global_hist_ms += std::chrono::duration<double, std::milli>(t_global_hist - t_histogram).count();
    timing.prefixsum_ms += std::chrono::duration<double, std::milli>(t_prefixsum - t_global_hist).count();
    timing.offsets_ms += std::chrono::duration<double, std::milli>(t_offsets - t_prefixsum).count();
    timing.scatter_ms += std::chrono::duration<double, std::milli>(t_scatter - t_offsets).count();
}

int main(int argc, char **argv) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Number of CUDA devices: " << deviceCount << "\n";
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Using GPU [0]: " << prop.name << "\n";
    }

    int64_t n = 10'000'000;
    if (argc > 1)
        n = atoll(argv[1]);
    std::cout << "Problem size: " << n << "\n";

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

    initRNG<<<blocks, TPB>>>(d_states, 42ULL, n);
    cudaDeviceSynchronize();
    std::cout << "Generating random values on GPU\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    generateRandom<<<blocks, TPB>>>(d_in_keys, d_in_vals, d_states, n);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gen_sec = t1 - t0;
    std::cout << "Generated random values in " << gen_sec.count() << " s on GPU\n";

    std::cout << "Sorting\n";
    auto t2 = std::chrono::high_resolution_clock::now();
    ShuffleTiming totalTiming;
    for (int digit = 0; digit < N_DIGITS; ++digit) {
        gpuShufflePerBlock<uint64_t>(d_in_keys, d_in_vals, d_out_keys, d_out_vals, d_counts, d_perBlockOffsets, n, digit, totalTiming);
        std::swap(d_in_keys, d_out_keys);
        std::swap(d_in_vals, d_out_vals);
    }
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sort_sec = t3 - t2;

    std::cout << "Sorted " << n << " values in " << sort_sec.count() << " s\n";
    double rate = double(n) / sort_sec.count() / 1e6;
    std::cout << "That's " << rate << " M elements sorted / s\n";

    std::cout << "Timing Breakdown (All Passes):\n";
    std::cout << "  Histogram: " << totalTiming.histogram_ms << " ms\n";
    std::cout << "  Global Hist: " << totalTiming.global_hist_ms << " ms\n";
    std::cout << "  Prefix Sum: " << totalTiming.prefixsum_ms << " ms\n";
    std::cout << "  Calculate Offsets: " << totalTiming.offsets_ms << " ms\n";
    std::cout << "  Stable Scatter: " << totalTiming.scatter_ms << " ms\n";

    std::vector<uint64_t> h_keys(n);
    cudaMemcpy(h_keys.data(), d_in_keys, sizeof(uint64_t) * n, cudaMemcpyDeviceToHost);
    thrust::device_vector<uint64_t> dv(h_keys.begin(), h_keys.end());
    bool is_sorted = thrust::is_sorted(dv.begin(), dv.end());
    assert(is_sorted);

    cudaFree(d_in_keys);
    cudaFree(d_out_keys);
    cudaFree(d_in_vals);
    cudaFree(d_out_vals);
    cudaFree(d_counts);
    cudaFree(d_perBlockOffsets);
    cudaFree(d_states);

    return 0;
}