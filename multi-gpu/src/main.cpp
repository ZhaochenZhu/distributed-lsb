#include <mpi.h>
#include <cuda_runtime.h>
#include "kernel_driver.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int worldSize, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // one GPU per MPI rank
  int devCount = 0;
  cudaGetDeviceCount(&devCount);
  cudaSetDevice(rank % devCount);

  // parse size
  int64_t N_global = 10'000'000;
  if (argc > 1) N_global = std::stoll(argv[1]);
  int64_t N_local  = N_global / worldSize;
  int blocks = (N_local + TPB - 1) / TPB;

  if (rank == 0) {
    std::cout << "Global N=" << N_global
              << ", per-rank N=" << N_local
              << ", ranks=" << worldSize << "\n";
  }

  // allocate device buffers
  uint64_t *d_in_keys, *d_in_vals, *d_out_keys, *d_out_vals;
  int *d_counts, *d_perBlockOffsets, *d_localCounts, *d_bucketStarts;

  ShuffleTiming timing{0.0, 0.0, 0.0, 0.0};

  cudaMalloc(&d_in_keys,        N_local * sizeof(uint64_t));
  cudaMalloc(&d_in_vals,        N_local * sizeof(uint64_t));
  cudaMalloc(&d_out_keys,       N_local * sizeof(uint64_t));
  cudaMalloc(&d_out_vals,       N_local * sizeof(uint64_t));
  cudaMalloc(&d_counts,         blocks * N_BUCKETS * sizeof(int));
  cudaMalloc(&d_perBlockOffsets,blocks * N_BUCKETS * sizeof(int));
  cudaMalloc(&d_localCounts,    N_BUCKETS * sizeof(int));
  cudaMalloc(&d_bucketStarts,   N_BUCKETS * sizeof(int));

  gpu_setup_rng(N_local);
  gpu_generate_random(d_in_keys, d_in_vals, N_local);

  MPI_Barrier(MPI_COMM_WORLD);
  auto t0 = std::chrono::high_resolution_clock::now();

  // LSD radix loop
  for (int digit = 0; digit < N_DIGITS; ++digit) {
    // local histogram + per-GPU reduction
    gpu_histogram_reduce(d_in_keys, d_counts, d_localCounts, digit, N_local, &timing);

    // global bucket totals
    std::vector<int> h_local(N_BUCKETS), h_global(N_BUCKETS);
    cudaMemcpy(h_local.data(), d_localCounts,
               N_BUCKETS * sizeof(int), cudaMemcpyDeviceToHost);
    MPI_Allreduce(h_local.data(), h_global.data(),
                  N_BUCKETS, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // per-rank prefix
    std::vector<int> h_prefix(N_BUCKETS);
    MPI_Exscan(h_local.data(), h_prefix.data(),
               N_BUCKETS, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) std::fill(h_prefix.begin(), h_prefix.end(), 0);

    // copy to device
    cudaMemcpy(d_bucketStarts, h_prefix.data(),
               N_BUCKETS * sizeof(int), cudaMemcpyHostToDevice);

    // per-block offsets
    gpu_compute_offsets(d_counts, d_bucketStarts,
                        d_perBlockOffsets, N_local, &timing);

    // scatter
    gpu_scatter(d_in_keys, d_in_vals,
                d_out_keys, d_out_vals,
                d_perBlockOffsets, N_local,
                digit, &timing);

    // swap
    std::swap(d_in_keys,  d_out_keys);
    std::swap(d_in_vals,  d_out_vals);
  }

  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_WORLD);
  auto t1 = std::chrono::high_resolution_clock::now();

  if (rank == 0) {
    double total_s = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Distributed radix sort time: " << total_s << " s\n"
              << "Timing (ms): hist=" << timing.histogram_ms
              << ", copy="    << timing.copyback_ms
              << ", off="     << timing.offsets_ms
              << ", scat="    << timing.scatter_ms << "\n";
  }

  // verify
  std::vector<uint64_t> hbuf(N_local);
  cudaMemcpy(hbuf.data(), d_in_keys,
             N_local * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  assert(std::is_sorted(hbuf.begin(), hbuf.end()));

  // cleanup
  cudaFree(d_in_keys);
  cudaFree(d_in_vals);
  cudaFree(d_out_keys);
  cudaFree(d_out_vals);
  cudaFree(d_counts);
  cudaFree(d_perBlockOffsets);
  cudaFree(d_localCounts);
  cudaFree(d_bucketStarts);

  MPI_Finalize();
  return 0;
}
