#include "kernel_driver.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <chrono>

namespace cg = cooperative_groups;
using clk = std::chrono::high_resolution_clock;

// Device kernels --------------------------------------------------

__global__ void computeLocalCounts(
    const uint64_t *keys,
    int *counts,
    int n,
    int digit
) {
    extern __shared__ int s_counts[];
    int lane = threadIdx.x;
    // zero shared histogram
    for(int b=lane; b<N_BUCKETS; b+=blockDim.x)
      s_counts[b] = 0;
    __syncthreads();

    size_t gid = blockIdx.x*blockDim.x + threadIdx.x;
    size_t stride = blockDim.x*gridDim.x;
    for(size_t i=gid; i<(size_t)n; i+=stride) {
      uint64_t key = keys[i];
      int bucket = (key >> (digit*RADIX)) & (N_BUCKETS-1);
      atomicAdd(&s_counts[bucket], 1);
    }
    __syncthreads();

    // write out per-block histogram
    for(int b=lane; b<N_BUCKETS; b+=blockDim.x)
      counts[blockIdx.x*N_BUCKETS + b] = s_counts[b];
}

__global__ void reduceHistogram(
    const int *counts,
    int *globalCounts,
    int numBlocks
) {
    int bucket = threadIdx.x;
    int sum = 0;
    for(int b=0; b<numBlocks; ++b)
      sum += counts[b*N_BUCKETS + bucket];
    globalCounts[bucket] = sum;
}

__global__ void computePerBlockOffsets(
    const int *counts,
    const int *starts,
    int *perBlockOffsets,
    int numBlocks
) {
    int bucket = blockIdx.x;
    int run = starts[bucket];
    for(int b=threadIdx.x; b<numBlocks; b+=blockDim.x) {
      int idx = b*N_BUCKETS + bucket;
      perBlockOffsets[idx] = run;
      run += counts[idx];
    }
}

__global__ void scatterParallel(
    const uint64_t *in_keys,
    const uint64_t *in_vals,
    uint64_t *out_keys,
    uint64_t *out_vals,
    const int *perBlockOffsets,
    int n,
    int digit
) {
    cg::thread_block tb = cg::this_thread_block();
    extern __shared__ int sh[];
    int *s_counts  = sh;
    int *s_offsets = sh + N_BUCKETS;

    int lane = threadIdx.x;
    int bid  = blockIdx.x;
    int start = bid*blockDim.x;
    int idx   = start + lane;

    // build block-local histogram in shared mem
    for(int b=lane; b<N_BUCKETS; b+=blockDim.x)
      s_counts[b] = 0;
    tb.sync();

    if(idx < n) {
      uint64_t key = in_keys[idx];
      int bucket = (key >> (digit*RADIX)) & (N_BUCKETS-1);
      atomicAdd(&s_counts[bucket], 1);
    }
    tb.sync();

    // exclusive prefix in shared mem
    if(lane < N_BUCKETS) {
      int sum = 0;
      for(int i=0; i<lane; ++i) sum += s_counts[i];
      s_offsets[lane] = sum;
    }
    tb.sync();

    // scatter each thread’s key into final slot
    if(idx < n) {
      uint64_t key = in_keys[idx];
      uint64_t val = in_vals[idx];
      int bucket = (key >> (digit*RADIX)) & (N_BUCKETS-1);

      // compute lane‑within‑bucket via ballot
      unsigned mask = __ballot_sync(0xFFFFFFFF,
                      (((key >> (digit*RADIX)) & (N_BUCKETS-1)) == bucket));
      int lane_id = __popc(mask & ((1u << (lane%32)) - 1));

      int base = perBlockOffsets[bid*N_BUCKETS + bucket];
      int dst  = base + s_offsets[bucket] + lane_id;
      out_keys[dst] = key;
      out_vals[dst] = val;
    }
}

__global__ void initRNG(curandState *states, unsigned long seed, int n) {
    size_t gid = blockIdx.x*blockDim.x + threadIdx.x;
    size_t stride = blockDim.x*gridDim.x;
    for(int i=gid; i<n; i+=stride)
      curand_init(seed, i, 0, &states[i]);
}

__global__ void generateRandom(
    uint64_t *d_keys,
    uint64_t *d_vals,
    curandState *states,
    int n
) {
    size_t gid = blockIdx.x*blockDim.x + threadIdx.x;
    size_t stride = blockDim.x*gridDim.x;
    for(int i=gid; i<n; i+=stride) {
      uint64_t r = ((uint64_t)curand(&states[i]) << 32)
                 |  (uint64_t)curand(&states[i]);
      d_keys[i] = r;
      d_vals[i] = i;
    }
}

// Static RNG buffer
static curandState *d_states = nullptr;

// C‑API wrappers --------------------------------------------------

extern "C" {

void gpu_setup_rng(int n) {
    int blocks = (n + TPB - 1)/TPB;
    if(!d_states)
      cudaMalloc(&d_states, sizeof(curandState)*n);
    initRNG<<<blocks, TPB>>>(d_states, 42ULL, n);
    cudaDeviceSynchronize();
}

void gpu_generate_random(uint64_t *d_keys, uint64_t *d_vals, int n) {
    int blocks = (n + TPB - 1)/TPB;
    generateRandom<<<blocks, TPB>>>(d_keys, d_vals, d_states, n);
    cudaDeviceSynchronize();
}

void gpu_histogram_reduce(
    const uint64_t *d_in_keys,
    int *d_counts,
    int *d_localCounts,
    int digit,
    int n,
    ShuffleTiming *timing
) {
    int blocks = (n + TPB - 1)/TPB;
    auto t0 = clk::now();
    computeLocalCounts<<<blocks, TPB, N_BUCKETS*sizeof(int)>>>(d_in_keys, d_counts, n, digit);
    cudaDeviceSynchronize();
    auto t1 = clk::now();
    timing->histogram_ms += std::chrono::duration<double,std::milli>(t1 - t0).count();

    reduceHistogram<<<1, N_BUCKETS>>>(d_counts, d_localCounts, blocks);
    cudaDeviceSynchronize();
    auto t2 = clk::now();
    timing->copyback_ms += std::chrono::duration<double,std::milli>(t2 - t1).count();
}

void gpu_compute_offsets(
    const int *d_counts,
    const int *d_bucketStarts,
    int *d_perBlockOffsets,
    int n,
    ShuffleTiming *timing
) {
    int blocks = (n + TPB - 1)/TPB;
    auto t0 = clk::now();
    computePerBlockOffsets<<<N_BUCKETS, TPB>>>(d_counts, d_bucketStarts, d_perBlockOffsets, blocks);
    cudaDeviceSynchronize();
    auto t1 = clk::now();
    timing->offsets_ms += std::chrono::duration<double,std::milli>(t1 - t0).count();
}

void gpu_scatter(
    const uint64_t *d_in_keys,
    const uint64_t *d_in_vals,
    uint64_t *d_out_keys,
    uint64_t *d_out_vals,
    const int *d_perBlockOffsets,
    int n,
    int digit,
    ShuffleTiming *timing
) {
    int blocks = (n + TPB - 1)/TPB;
    size_t shmem = 2 * N_BUCKETS * sizeof(int);
    auto t0 = clk::now();
    scatterParallel<<<blocks, TPB, shmem>>>(d_in_keys, d_in_vals, d_out_keys, d_out_vals,
                                            d_perBlockOffsets, n, digit);
    cudaDeviceSynchronize();
    auto t1 = clk::now();
    timing->scatter_ms += std::chrono::duration<double,std::milli>(t1 - t0).count();
}

} // extern "C"
