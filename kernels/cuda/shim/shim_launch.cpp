// shim_launch.cpp -- builds sumfold_shim.dll: the unmodified kernels/cuda/sumfold.cu executed on the CPU through cuda_host_shim.h, with one exported launcher per kernel. Used by bench/runners/kernel_gpu_bench.py (backend "shim") and tests/test_kernels_gpu_emu.py.
//
// Build (MinGW-w64 g++ 6.2, see kernels/cuda/README.md): g++ -O2 -std=c++14 -fopenmp -shared -static-libgcc -static-libstdc++ -Wno-unknown-pragmas -o kernels/cuda/build/sumfold_shim.dll kernels/cuda/shim/shim_launch.cpp
#include "cuda_host_shim.h"

#include <cstdlib>

thread_local shim_dim3 threadIdx;
shim_dim3 blockIdx, blockDim, gridDim;
unsigned long long shim_shfl_buf[1024];

#include "../sumfold.cu"

#define SHIM_EXPORT extern "C" __declspec(dllexport)

static int g_order = 0;        // 0 ascending, 1 descending, 2 odd blocks first then even
static int g_bad_team = 0;     // set if OpenMP did not deliver blockDim.x threads

static unsigned block_at(unsigned i, unsigned grid) {
    if (g_order == 1) return grid - 1 - i;
    if (g_order == 2) {
        unsigned nodd = grid / 2;
        return i < nodd ? 2 * i + 1 : 2 * (i - nodd);
    }
    return i;
}

template <class F>
static int launch(unsigned grid, unsigned block, F body) {
    if (block == 0 || block > 1024 || (block & 31u) != 0 || grid == 0) return -1;
    omp_set_dynamic(0);
    gridDim = {grid, 1, 1};
    blockDim = {block, 1, 1};
    for (unsigned i = 0; i < grid; i++) {
        blockIdx = {block_at(i, grid), 0, 0};
#pragma omp parallel num_threads(block)
        {
            threadIdx = {(unsigned)omp_get_thread_num(), 0, 0};
            if ((unsigned)omp_get_num_threads() != block) {
#pragma omp atomic write
                g_bad_team = 1;
            } else {
                body();
            }
        }
        if (g_bad_team) return -2;
    }
    return 0;
}

SHIM_EXPORT int shim_set_block_order(int order) {
    g_order = order;
    return 0;
}

SHIM_EXPORT int shim_sf_eval(unsigned grid, unsigned block, const ext_t *f, const ext_t *g,
                             long long npairs, ext_t *partials) {
    return launch(grid, block, [&] { sf_eval(f, g, npairs, partials); });
}

SHIM_EXPORT int shim_sf_fold_eval(unsigned grid, unsigned block, const ext_t *fs, const ext_t *gs,
                                  ext_t *fd, ext_t *gd, long long nquads, u64 r0, u64 r1,
                                  ext_t *partials) {
    return launch(grid, block, [&] { sf_fold_eval(fs, gs, fd, gd, nquads, r0, r1, partials); });
}

SHIM_EXPORT int shim_sf_fold(unsigned grid, unsigned block, const ext_t *fs, const ext_t *gs,
                             ext_t *fd, ext_t *gd, long long npairs, u64 r0, u64 r1) {
    return launch(grid, block, [&] { sf_fold(fs, gs, fd, gd, npairs, r0, r1); });
}

SHIM_EXPORT int shim_sf_reduce_partials(unsigned grid, unsigned block, const ext_t *partials,
                                        int nblocks, ext_t *out3) {
    return launch(grid, block, [&] { sf_reduce_partials(partials, nblocks, out3); });
}
