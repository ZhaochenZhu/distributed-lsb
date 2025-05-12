// File: radix_sort_ppsum_parallel_scatter.cu

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <cassert>
#include <cooperative_groups.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

namespace cg = cooperative_groups;

#define RADIX 8
#define N_BUCKETS (1 << RADIX)
#define N_DIGITS (64 / RADIX)
#define TPB 256

struct ShuffleTiming {
    double histogram_ms = 0;
    double copyback_ms = 0;
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

__global__ void reduceHistogram(const int *counts, int *global, int numBlocks) {
    int bucket = threadIdx.x;
    int sum = 0;
    for (int b = 0; b < numBlocks; ++b)
        sum += counts[b * N_BUCKETS + bucket];
    global[bucket] = sum;
}

__global__ void computePerBlockOffsets(const int *counts, const int *starts, int *perBlockOffsets, int numBlocks) {
    int bucket = blockIdx.x;
    int running = starts[bucket];
    for (int b = threadIdx.x; b < numBlocks; b += blockDim.x) {
        int idx = b * N_BUCKETS + bucket;
        perBlockOffsets[idx] = running;
        running += counts[idx];
    }
}

// Parallel scatter using cooperative groups and shared memory
__global__ void scatterParallel(const uint64_t *in_keys,
                                const uint64_t *in_vals,
                                uint64_t *out_keys,
                                uint64_t *out_vals,
                                const int *perBlockOffsets,
                                int n, int digit) {
    cg::thread_block tb = cg::this_thread_block();
    extern __shared__ int shmem[];        // 2 * N_BUCKETS ints
    int *s_counts  = shmem;
    int *s_offsets = shmem + N_BUCKETS;

    int lane = threadIdx.x;
    int bidx = blockIdx.x;
    int start = bidx * blockDim.x;
    int idx = start + lane;

    // 1) initialize shared counts
    for (int b = lane; b < N_BUCKETS; b += blockDim.x)
        s_counts[b] = 0;
    tb.sync();

    // 2) local histogram
    if (idx < n) {
        uint64_t key = in_keys[idx];
        int bucket = (key >> (digit * RADIX)) & (N_BUCKETS - 1);
        atomicAdd(&s_counts[bucket], 1);
    }
    tb.sync();

    // 3) exclusive prefix sum on histogram
    if (lane < N_BUCKETS) {
        int sum = 0;
        for (int i = 0; i < lane; ++i)
            sum += s_counts[i];
        s_offsets[lane] = sum;
    }
    tb.sync();

    // 4) each thread scatter its element
    if (idx < n) {
        uint64_t key = in_keys[idx];
        uint64_t val = in_vals[idx];
        int bucket = (key >> (digit * RADIX)) & (N_BUCKETS - 1);

        // warp-level index within bucket
        unsigned mask = __ballot_sync(0xffffffff, 
                            ((key >> (digit * RADIX)) & (N_BUCKETS - 1)) == bucket);
        int lane_id = __popc(mask & ((1u << (lane % 32)) - 1));

        int base = perBlockOffsets[bidx * N_BUCKETS + bucket];
        int dst = base + s_offsets[bucket] + lane_id;
        out_keys[dst] = key;
        out_vals[dst] = val;
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
void gpuShufflePerBlock(const T *d_in_keys, const T *d_in_vals, T *d_out_keys, T *d_out_vals,
                        int *d_counts, int *d_perBlockOffsets, int *d_global, int *d_starts,
                        int n, int digit, int blocks, ShuffleTiming &timing) {
    using clk = std::chrono::high_resolution_clock;

    auto t_start = clk::now();
    computeLocalCounts<<<blocks, TPB, N_BUCKETS * sizeof(int)>>>(d_in_keys, d_counts, n, digit);
    cudaDeviceSynchronize();
    auto t_hist = clk::now();

    reduceHistogram<<<1, N_BUCKETS>>>(d_counts, d_global, blocks);
    cudaDeviceSynchronize();
    auto t_copy = clk::now();

    thrust::device_ptr<int> gptr(d_global);
    thrust::device_ptr<int> sptr(d_starts);
    thrust::exclusive_scan(thrust::device, gptr, gptr + N_BUCKETS, sptr);
    cudaDeviceSynchronize();
    auto t_psum = clk::now();

    computePerBlockOffsets<<<N_BUCKETS, TPB>>>(d_counts, d_starts, d_perBlockOffsets, blocks);
    cudaDeviceSynchronize();
    auto t_off = clk::now();

    // scatter in parallel
    size_t shmem_bytes = 2 * N_BUCKETS * sizeof(int);
    scatterParallel<<<blocks, TPB, shmem_bytes>>>(d_in_keys, d_in_vals, d_out_keys, d_out_vals,
                                                 d_perBlockOffsets, n, digit);
    cudaDeviceSynchronize();
    auto t_scat = clk::now();

    timing.histogram_ms += std::chrono::duration<double, std::milli>(t_hist - t_start).count();
    timing.copyback_ms  += std::chrono::duration<double, std::milli>(t_copy - t_hist).count();
    timing.prefixsum_ms += std::chrono::duration<double, std::milli>(t_psum - t_copy).count();
    timing.offsets_ms   += std::chrono::duration<double, std::milli>(t_off - t_psum).count();
    timing.scatter_ms   += std::chrono::duration<double, std::milli>(t_scat - t_off).count();
}

int main(int argc, char **argv) {
    int64_t n = 10'000'000;
    if (argc > 1) n = atoll(argv[1]);
    std::cout << "Problem size: " << n << "\n";

    uint64_t *d_in_keys, *d_out_keys;
    uint64_t *d_in_vals, *d_out_vals;
    int *d_counts, *d_perBlockOffsets;
    int *d_global, *d_starts;
    curandState *d_states;

    int blocks = (n + TPB - 1) / TPB;

    cudaMalloc(&d_in_keys, sizeof(uint64_t) * n);
    cudaMalloc(&d_out_keys, sizeof(uint64_t) * n);
    cudaMalloc(&d_in_vals, sizeof(uint64_t) * n);
    cudaMalloc(&d_out_vals, sizeof(uint64_t) * n);
    cudaMalloc(&d_counts, sizeof(int) * blocks * N_BUCKETS);
    cudaMalloc(&d_perBlockOffsets, sizeof(int) * blocks * N_BUCKETS);
    cudaMalloc(&d_global, sizeof(int) * N_BUCKETS);
    cudaMalloc(&d_starts, sizeof(int) * N_BUCKETS);
    cudaMalloc(&d_states, sizeof(curandState) * n);

    initRNG<<<blocks, TPB>>>(d_states, 42ULL, n);
    cudaDeviceSynchronize();

    std::cout << "Generating random values on GPU\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    generateRandom<<<blocks, TPB>>>(d_in_keys, d_in_vals, d_states, n);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Generated random values in " << std::chrono::duration<double>(t1 - t0).count() << " s\n";

    std::cout << "Sorting\n";
    auto t2 = std::chrono::high_resolution_clock::now();
    ShuffleTiming totalTiming;
    for (int digit = 0; digit < N_DIGITS; ++digit) {
        gpuShufflePerBlock<uint64_t>(
            d_in_keys, d_in_vals, d_out_keys, d_out_vals,
            d_counts, d_perBlockOffsets, d_global, d_starts,
            n, digit, blocks, totalTiming);
        std::swap(d_in_keys, d_out_keys);
        std::swap(d_in_vals, d_out_vals);
    }
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    std::cout << "Sorted " << n << " values in " << std::chrono::duration<double>(t3 - t2).count() << " s\n";

    std::cout << "Timing Breakdown (All Passes):\n";
    std::cout << "  Histogram: " << totalTiming.histogram_ms << " ms\n";
    std::cout << "  Copyback: " << totalTiming.copyback_ms << " ms\n";
    std::cout << "  Prefix Sum: " << totalTiming.prefixsum_ms << " ms\n";
    std::cout << "  Calculate Offsets: " << totalTiming.offsets_ms << " ms\n";
    std::cout << "  Stable Scatter: " << totalTiming.scatter_ms << " ms\n";

    std::vector<uint64_t> h_keys(n);
    cudaMemcpy(h_keys.data(), d_in_keys, sizeof(uint64_t) * n, cudaMemcpyDeviceToHost);
    thrust::device_vector<uint64_t> dv(h_keys.begin(), h_keys.end());
    assert(thrust::is_sorted(dv.begin(), dv.end()));

    cudaFree(d_in_keys);
    cudaFree(d_out_keys);
    cudaFree(d_in_vals);
    cudaFree(d_out_vals);
    cudaFree(d_counts);
    cudaFree(d_perBlockOffsets);
    cudaFree(d_global);
    cudaFree(d_starts);
    cudaFree(d_states);

    return 0;
}
