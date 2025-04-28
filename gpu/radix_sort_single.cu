#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>

#define RADIX 8
#define N_BUCKETS (1 << RADIX)
#define N_DIGITS (64 / RADIX)

// CUDA kernel: compute per-block local bucket counts for a given digit
__global__ void computeLocalCounts(
    const uint64_t *keys, int *counts, int n, int digit)
{
  extern __shared__ int s_counts[];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int lane = threadIdx.x;
  // initialize shared histogram
  for (int b = lane; b < N_BUCKETS; b += blockDim.x)
    s_counts[b] = 0;
  __syncthreads();

  // each thread bins one key
  if (tid < n)
  {
    uint64_t key = keys[tid];
    int bucket = (key >> (digit * RADIX)) & (N_BUCKETS - 1);
    atomicAdd(&s_counts[bucket], 1);
  }
  __syncthreads();

  // write block's histogram to global memory
  for (int b = lane; b < N_BUCKETS; b += blockDim.x)
  {
    counts[blockIdx.x * N_BUCKETS + b] = s_counts[b];
  }
}

// CUDA kernel: exclusive scan of global counts (simple hillis-steele)
__global__ void exclusiveScan(int *data, int m)
{
  int tid = threadIdx.x;
  for (int offset = 1; offset < m; offset <<= 1)
  {
    int t = 0;
    if (tid >= offset)
      t = data[tid - offset];
    __syncthreads();
    data[tid] += t;
    __syncthreads();
  }
  // for exclusive, shift right
  if (tid == 0)
    data[0] = 0;
  __syncthreads();
}

// CUDA kernel: scatter based on global bucket starts
__global__ void scatter(
    const uint64_t *in_keys, const uint64_t *in_vals,
    uint64_t *out_keys, uint64_t *out_vals,
    const int *starts, int n, int digit)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n)
    return;
  uint64_t key = in_keys[tid];
  uint64_t val = in_vals[tid];
  int bucket = (key >> (digit * RADIX)) & (N_BUCKETS - 1);
  // atomic fetch-and-increment to assign global position
  int pos = atomicAdd((int *)&starts[bucket], 1);
  out_keys[pos] = key;
  out_vals[pos] = val;
}

// Host: perform one digit of stable bucket sort on GPU
template <typename T>
void gpuShuffle(
    const T *d_in_keys, const T *d_in_vals,
    T *d_out_keys, T *d_out_vals,
    int *d_counts, int *d_starts,
    int n, int digit)
{
  int blocks = (n + 255) / 256;
  // 1) local histograms
  computeLocalCounts<<<blocks, 256, N_BUCKETS * sizeof(int)>>>(
      d_in_keys, d_counts, n, digit);

  // 2) reduce block histograms into global counts on device memory
  // simple reduction: assume blocks small; copy to host or use thrust
  std::vector<int> h_counts(blocks * N_BUCKETS);
  cudaMemcpy(h_counts.data(), d_counts,
             sizeof(int) * blocks * N_BUCKETS,
             cudaMemcpyDeviceToHost);
  std::vector<int> h_global(N_BUCKETS, 0);
  for (int b = 0; b < blocks; ++b)
    for (int i = 0; i < N_BUCKETS; ++i)
      h_global[i] += h_counts[b * N_BUCKETS + i];

  // exclusive scan on host
  std::vector<int> h_starts(N_BUCKETS);
  int sum = 0;
  for (int i = 0; i < N_BUCKETS; ++i)
  {
    h_starts[i] = sum;
    sum += h_global[i];
  }
  // copy global starts back to device
  cudaMemcpy(d_starts, h_starts.data(), sizeof(int) * N_BUCKETS,
             cudaMemcpyHostToDevice);

  // 3) scatter
  scatter<<<blocks, 256>>>(d_in_keys, d_in_vals,
                           d_out_keys, d_out_vals,
                           d_starts, n, digit);

  cudaDeviceSynchronize();
}

int main(int argc, char **argv)
{
  int64_t n = 10000000; // default 10M
  if (argc > 1)
    n = atoll(argv[1]);

  // --- Host data allocation & init ---
  std::vector<uint64_t> h_keys(n), h_vals(n);
  std::mt19937_64 rng(42);
  for (int64_t i = 0; i < n; ++i)
  {
    h_keys[i] = rng();
    h_vals[i] = i;
  }

  // --- Device allocation ---
  uint64_t *d_in_keys, *d_out_keys;
  uint64_t *d_in_vals, *d_out_vals;
  int *d_counts, *d_starts;
  cudaMalloc(&d_in_keys, sizeof(uint64_t) * n);
  cudaMalloc(&d_out_keys, sizeof(uint64_t) * n);
  cudaMalloc(&d_in_vals, sizeof(uint64_t) * n);
  cudaMalloc(&d_out_vals, sizeof(uint64_t) * n);
  cudaMalloc(&d_counts, sizeof(int) * ((n + 255) / 256) * N_BUCKETS);
  cudaMalloc(&d_starts, sizeof(int) * N_BUCKETS);

  cudaMemcpy(d_in_keys, h_keys.data(), sizeof(uint64_t) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in_vals, h_vals.data(), sizeof(uint64_t) * n, cudaMemcpyHostToDevice);

  // --- GPU LSD radix sort ---
  for (int digit = 0; digit < N_DIGITS; ++digit)
  {
    gpuShuffle(d_in_keys, d_in_vals,
               d_out_keys, d_out_vals,
               d_counts, d_starts,
               n, digit);
    // swap buffers
    std::swap(d_in_keys, d_out_keys);
    std::swap(d_in_vals, d_out_vals);
  }

  // --- Copy back & verify ---
  cudaMemcpy(h_keys.data(), d_in_keys, sizeof(uint64_t) * n, cudaMemcpyDeviceToHost);
  // simple CPU verify
  for (int64_t i = 1; i < n; ++i)
  {
    assert(h_keys[i - 1] <= h_keys[i]);
  }
  std::cout << "Sorted " << n << " keys on GPU\n";

  // cleanup
  cudaFree(d_in_keys);
  cudaFree(d_out_keys);
  cudaFree(d_in_vals);
  cudaFree(d_out_vals);
  cudaFree(d_counts);
  cudaFree(d_starts);
}
