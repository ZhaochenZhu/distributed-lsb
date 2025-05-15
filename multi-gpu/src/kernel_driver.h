#ifndef KERNEL_DRIVER_H
#define KERNEL_DRIVER_H

#include <cstdint>

#define RADIX      8
#define N_BUCKETS  (1 << RADIX)
#define N_DIGITS   (64 / RADIX)
#define TPB        256

struct ShuffleTiming {
    double histogram_ms;
    double copyback_ms;
    double offsets_ms;
    double scatter_ms;
};

#ifdef __cplusplus
extern "C" {
#endif

void gpu_setup_rng(int n);
void gpu_generate_random(uint64_t *d_keys, uint64_t *d_vals, int n);

void gpu_histogram_reduce(
    const uint64_t *d_in_keys,
    int *d_counts,
    int *d_localCounts,
    int digit,
    int n,
    ShuffleTiming *timing
);

// per-block offsets on device
void gpu_compute_offsets(
    const int *d_counts,
    const int *d_bucketStarts,
    int *d_perBlockOffsets,
    int n,
    ShuffleTiming *timing
);

// scatter into output buffer
void gpu_scatter(
    const uint64_t *d_in_keys,
    const uint64_t *d_in_vals,
    uint64_t *d_out_keys,
    uint64_t *d_out_vals,
    const int *d_perBlockOffsets,
    int n,
    int digit,
    ShuffleTiming *timing
);

#ifdef __cplusplus
}
#endif

#endif // KERNEL_DRIVER_H
