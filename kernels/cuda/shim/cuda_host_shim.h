// cuda_host_shim.h -- minimal SIMT host emulation so that kernels/cuda/sumfold.cu can be compiled by g++ and *executed* on the CPU (no CUDA toolkit on this host).
//
// Model: one CUDA block = one OpenMP team of blockDim.x threads (threads really run concurrently); blocks of a grid run one after another in a caller-chosen order.
//   __syncthreads()      -> OpenMP barrier of the team
//   __shfl_down_sync()   -> exchange through a block-wide buffer between two barriers
// (valid because sumfold.cu calls every shuffle with uniform control flow across the whole block, and blockDim.x % 32 == 0)
//   __shared__           -> function-static storage (one block alive at a time)
//   __umul64hi           -> unsigned __int128 product
// What this shim can NOT show: real GPU timing, memory-model effects between concurrently running blocks, or NVRTC/ptxas acceptance of the source.
#ifndef EDGECERT_CUDA_HOST_SHIM_H
#define EDGECERT_CUDA_HOST_SHIM_H

#include <omp.h>
#include <cstdint>

struct shim_dim3 {
    unsigned x, y, z;
};

extern thread_local shim_dim3 threadIdx;
extern shim_dim3 blockIdx, blockDim, gridDim;
extern unsigned long long shim_shfl_buf[1024];

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static
#define __align__(n) alignas(n)

inline void __syncthreads() {
#pragma omp barrier
}

inline unsigned long long __umul64hi(unsigned long long a, unsigned long long b) {
    return (unsigned long long)(((unsigned __int128)a * b) >> 64);
}

inline unsigned long long __shfl_down_sync(unsigned mask, unsigned long long v, int off) {
    (void)mask;
    const unsigned tid = threadIdx.x, lane = tid & 31u, base = tid - lane;
    shim_shfl_buf[tid] = v;
#pragma omp barrier
    unsigned long long r = v;
    if (lane + (unsigned)off < 32u && base + lane + (unsigned)off < blockDim.x)
        r = shim_shfl_buf[base + lane + off];
#pragma omp barrier
    return r;
}

#endif
