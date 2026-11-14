// sumfold.cu -- CUDA SumFold kernels (product sumcheck over Goldilocks / Ext2).
//
// Semantics: spec/sumfold_kernel.md v1, bit-exact with src/edgecert/sumfold.py::prove(). Device-only source (no host code, no includes) so that it compiles unchanged with
//   * NVRTC through CuPy RawModule (bench/runners/kernel_gpu_bench.py, backend "cupy"), and
//   * g++ through the SIMT host shim (kernels/cuda/shim/), which executes these exact kernels on the CPU for bit-exactness tests when no CUDA toolkit is available.
//
// Data layout: one Ext element = 16 bytes {c0 (low), c1 (high)} (struct ext_t, 16-byte aligned so loads/stores are single 128-bit transactions), arrays of N elements.
//
// Kernels (launch with blockDim.x a multiple of 32, <= 1024; every thread of a block must reach block_reduce_store, so no thread returns early):
//   sf_eval            evaluate pass: round message partials of one table level
//   sf_fold_eval       FUSED pass: fold level i-1 with r_{i-1} -> level i (dst) and
//                      accumulate the round-i message in the same sweep
//   sf_fold            fold-only pass (final fold of the fused variant; fold pass of the
// unfused variant) sf_reduce_partials one block: sums the per-block partials -> 3 Ext words
//
// Why ping-pong instead of in-place on the GPU.  The RTL and the CPU kernel fold in place because the write index j never overtakes the read index 2j *within one sequential stream* (one lane).  A GPU grid has no such order: with a grid-stride loop, block b writes f'[j] for j in its own output range, which is the input range of block ~b/2; blocks are scheduled in no guaranteed order (and a grid-stride loop revisits the array several times), so a high-numbered block may overwrite inputs a low-numbered block has not read yet.  A grid-wide barrier between "read" and "write" would need cooperative launch, and a lane-local in-place scheme would pin one block per lane and add a host tail.  The kernels therefore read level i-1 from 'src' and write level i to 'dst' (ping-pong: level 0 in A[N], level 1 in B[N/2], level 2 in A[0..N/4), ...), i.e. 1.5x the resident table memory of the in-place design; an emulator run with reversed block order shows the in-place variant producing wrong transcripts (bench/runners/kernel_cuda_emu.py, tests/test_kernels_gpu_emu.py).
//
// Arithmetic: 64x64->128 products via a*b and __umul64hi(a, b); Goldilocks reduction exactly as field.reduce128 (lo - hh + hl*(2^32-1), multiplier-free, branch-free corrections); Ext2 Karatsuba (3 base products + 7*t1 as (t1<<3) - t1). Round sums use exact 128-bit lazy accumulators per thread, reduced once, then field-add warp shuffles and a shared-memory stage; field addition is exact, so the reduction tree shape never changes a bit of the output.

typedef unsigned long long u64;

#define GL_P   0xFFFFFFFF00000001ULL
#define GL_EPS 0x00000000FFFFFFFFULL
#define GL_MASK(c) (0ULL - (u64)(c))
#define SF_MAX_WARPS 32

struct __align__(16) ext_t {
    u64 c0;
    u64 c1;
};

// ------------------------------------------------------------ base field
__device__ __forceinline__ u64 gl_add(u64 a, u64 b) {
    u64 s = a + b;
    s += GL_EPS & GL_MASK(s < a);       // carry out of 2^64: s + 2^64 - p = s + EPS
    s -= GL_P & GL_MASK(s >= GL_P);
    return s;
}

__device__ __forceinline__ u64 gl_sub(u64 a, u64 b) {
    u64 d = a - b;
    d -= GL_EPS & GL_MASK(a < b);       // borrow: + p == - EPS (mod 2^64)
    return d;
}

__device__ __forceinline__ u64 gl_reduce128(u64 lo, u64 hi) {
    const u64 hh = hi >> 32;
    const u64 hl = hi & 0xFFFFFFFFULL;
    u64 t = lo - hh;
    t -= GL_EPS & GL_MASK(lo < hh);
    const u64 u = (hl << 32) - hl;      // hl * (2^32 - 1)
    u64 s = t + u;
    s += GL_EPS & GL_MASK(s < t);
    s -= GL_P & GL_MASK(s >= GL_P);
    return s;
}

__device__ __forceinline__ u64 gl_mul(u64 a, u64 b) {
    return gl_reduce128(a * b, __umul64hi(a, b));
}

__device__ __forceinline__ u64 gl_mul7(u64 t) {
    return gl_sub(gl_reduce128(t << 3, t >> 61), t);
}

// ------------------------------------------------------------- extension
__device__ __forceinline__ ext_t ext_make(u64 c0, u64 c1) {
    ext_t r;
    r.c0 = c0;
    r.c1 = c1;
    return r;
}

__device__ __forceinline__ ext_t ext_add(ext_t a, ext_t b) {
    return ext_make(gl_add(a.c0, b.c0), gl_add(a.c1, b.c1));
}

__device__ __forceinline__ ext_t ext_sub(ext_t a, ext_t b) {
    return ext_make(gl_sub(a.c0, b.c0), gl_sub(a.c1, b.c1));
}

__device__ __forceinline__ ext_t ext_mul(ext_t a, ext_t b) {
    const u64 t0 = gl_mul(a.c0, b.c0);
    const u64 t1 = gl_mul(a.c1, b.c1);
    const u64 t2 = gl_mul(gl_add(a.c0, a.c1), gl_add(b.c0, b.c1));
    return ext_make(gl_add(t0, gl_mul7(t1)), gl_sub(gl_sub(t2, t0), t1));
}

__device__ __forceinline__ ext_t ext_fold(ext_t v0, ext_t v1, ext_t r) {
    return ext_add(v0, ext_mul(r, ext_sub(v1, v0)));
}

// ------------------------------------------------------ lazy accumulators
struct acc6_t {
    u64 lo[6];
    u64 hi[6];
};

__device__ __forceinline__ void acc_zero(acc6_t &A) {
#pragma unroll
    for (int k = 0; k < 6; k++) {
        A.lo[k] = 0ULL;
        A.hi[k] = 0ULL;
    }
}

__device__ __forceinline__ void acc_add(acc6_t &A, int k, u64 x) {
    const u64 s = A.lo[k] + x;
    A.hi[k] += (s < x) ? 1ULL : 0ULL;
    A.lo[k] = s;
}

__device__ __forceinline__ void eval_pair(acc6_t &A, ext_t f0, ext_t f1, ext_t g0, ext_t g1) {
    const ext_t p0 = ext_mul(f0, g0);
    const ext_t p1 = ext_mul(f1, g1);
    const ext_t p2 = ext_mul(ext_sub(ext_add(f1, f1), f0), ext_sub(ext_add(g1, g1), g0));
    acc_add(A, 0, p0.c0);
    acc_add(A, 1, p0.c1);
    acc_add(A, 2, p1.c0);
    acc_add(A, 3, p1.c1);
    acc_add(A, 4, p2.c0);
    acc_add(A, 5, p2.c1);
}

// ------------------------------------------------------ block reduction
__device__ __forceinline__ u64 warp_reduce_field(u64 v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
                // lanes whose source lane is >= 32 receive their own value; only lane 0's result is used, and it only ever combines in-range lanes.
        v = gl_add(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    return v;
}

// Reduce the block's 6 accumulators to 3 Ext words at out[3*blockIdx.x + k]. Must be called by every thread of the block (uniform control flow): both warp-level stages are executed by all warps, so every __shfl_down_sync sees a full warp.
__device__ void block_reduce_store(acc6_t &A, ext_t *out) {
    __shared__ u64 smem[SF_MAX_WARPS][6];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    u64 v[6];
#pragma unroll
    for (int k = 0; k < 6; k++) v[k] = warp_reduce_field(gl_reduce128(A.lo[k], A.hi[k]));
    if (lane == 0) {
#pragma unroll
        for (int k = 0; k < 6; k++) smem[warp][k] = v[k];
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 6; k++) v[k] = warp_reduce_field(lane < nwarps ? smem[lane][k] : 0ULL);
    if (threadIdx.x == 0) {
        out[3 * blockIdx.x + 0] = ext_make(v[0], v[1]);
        out[3 * blockIdx.x + 1] = ext_make(v[2], v[3]);
        out[3 * blockIdx.x + 2] = ext_make(v[4], v[5]);
    }
}

// ------------------------------------------------------------- kernels
// Evaluate pass over a level with 2*npairs elements.
extern "C" __global__ void sf_eval(const ext_t *__restrict__ f, const ext_t *__restrict__ g,
                                   long long npairs, ext_t *__restrict__ partials) {
    acc6_t A;
    acc_zero(A);
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x; j < npairs; j += stride) {
        eval_pair(A, f[2 * j], f[2 * j + 1], g[2 * j], g[2 * j + 1]);
    }
    block_reduce_store(A, partials);
}

// FUSED pass i >= 1: src holds level i-1 (4*nquads elements), dst receives level i (2*nquads elements); the round-i message is accumulated from the folded outputs.
extern "C" __global__ void sf_fold_eval(const ext_t *__restrict__ fs, const ext_t *__restrict__ gs,
                                        ext_t *__restrict__ fd, ext_t *__restrict__ gd,
                                        long long nquads, u64 r0, u64 r1,
                                        ext_t *__restrict__ partials) {
    const ext_t r = ext_make(r0, r1);
    acc6_t A;
    acc_zero(A);
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; k < nquads; k += stride) {
        const ext_t a0 = ext_fold(fs[4 * k], fs[4 * k + 1], r);
        const ext_t a1 = ext_fold(fs[4 * k + 2], fs[4 * k + 3], r);
        const ext_t b0 = ext_fold(gs[4 * k], gs[4 * k + 1], r);
        const ext_t b1 = ext_fold(gs[4 * k + 2], gs[4 * k + 3], r);
        fd[2 * k] = a0;
        fd[2 * k + 1] = a1;
        gd[2 * k] = b0;
        gd[2 * k + 1] = b1;
        eval_pair(A, a0, a1, b0, b1);
    }
    block_reduce_store(A, partials);
}

// Fold-only pass: dst[j] = src[2j] + r*(src[2j+1] - src[2j]) for j < npairs.
extern "C" __global__ void sf_fold(const ext_t *__restrict__ fs, const ext_t *__restrict__ gs,
                                   ext_t *__restrict__ fd, ext_t *__restrict__ gd,
                                   long long npairs, u64 r0, u64 r1) {
    const ext_t r = ext_make(r0, r1);
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x; j < npairs; j += stride) {
        fd[j] = ext_fold(fs[2 * j], fs[2 * j + 1], r);
        gd[j] = ext_fold(gs[2 * j], gs[2 * j + 1], r);
    }
}

// Launch with exactly one block: out3[0..2] = sum over b < nblocks of partials[3b..3b+2].
extern "C" __global__ void sf_reduce_partials(const ext_t *__restrict__ partials, int nblocks,
                                              ext_t *__restrict__ out3) {
    acc6_t A;
    acc_zero(A);
    for (int b = threadIdx.x; b < nblocks; b += blockDim.x) {
#pragma unroll
        for (int k = 0; k < 3; k++) {
            const ext_t e = partials[3 * b + k];
            acc_add(A, 2 * k, e.c0);
            acc_add(A, 2 * k + 1, e.c1);
        }
    }
    block_reduce_store(A, out3);  // blockIdx.x == 0 -> out3[0..2]
}
