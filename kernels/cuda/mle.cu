// mle.cu -- CUDA MLE partial evaluation (batched segment folds) and a device-resident, batched, transcript-bound product sumcheck for the TRANSPARENT_REFERENCE matmul prover (src/edgecert/prover_adapters/matmul_sumcheck.py, GpuBackend).
//
// NOT standalone: compiled by NVRTC as  sumfold.cu + "\n" + mle.cu  (one CuPy RawModule), and
// reuses from kernels/cuda/sumfold.cu: u64, ext_t, GL_P/GL_EPS/GL_MASK, gl_add/gl_sub/gl_mul, ext_make/ext_add/ext_sub/ext_mul/ext_fold, acc6_t/acc_zero/acc_add, eval_pair and block_reduce_store.  Semantics are those of src/edgecert/sumfold.py (golden model) and edgecert.crypto.transcript.BoundSumcheckTranscript; outputs are bit-exact with the CPU library kernels/cpu/mle.c and the pure-Python reference (tests/test_matmul_sumcheck*.py).
//
// Kernels mle_fold_i8 / mle_fold_u16 / mle_fold_ext one fold pass of up to 4 levels: every thread folds one aligned segment of 2^nlev entries of the fold axis completely in registers (first level base x ext for integer input), for a batch of independent instances (blockIdx.y), each with its own challenges.  A full partial evaluation = ceil(nr / 4) passes (host loop). mm_sumcheck one thread block per instance runs all rounds of the product sumcheck: pass 0 evaluate, fused fold+evaluate passes (ping-pong scratch, level 0 is read-only and zero-extended beyond m0), final fold; thread 0 runs the SHA-256 transcript on the device (golden init, binding hash, rounds), so there is no host round trip per round.

#define MLE_SEG_LOG 4
#define MLE_SEG (1 << MLE_SEG_LOG)
#define MLE_MAX_ROUNDS 40
#define MLE_MAX_LABEL 96

__device__ __forceinline__ u64 mle_lift(long long v) { return (u64)v + (GL_P & GL_MASK(v < 0)); }

// fold(embed(a), embed(b), r) = (a + r0 d, r1 d), d = b - a
__device__ __forceinline__ ext_t mle_base_fold(u64 a, u64 b, ext_t r) {
    const u64 d = gl_sub(b, a);
    return ext_make(gl_add(a, gl_mul(r.c0, d)), gl_mul(r.c1, d));
}

__device__ __forceinline__ ext_t mle_load(const signed char *X, long long idx) {
    return ext_make(mle_lift((long long)X[idx]), 0ULL);
}
__device__ __forceinline__ ext_t mle_load(const unsigned short *X, long long idx) {
    return ext_make((u64)X[idx], 0ULL);
}
__device__ __forceinline__ ext_t mle_load(const ext_t *X, long long idx) { return X[idx]; }

// Instance b = blockIdx.y: X_b = in + b*in_bstride, element (i, k) at X_b[i*sf + k*sk], i < nfold, k < nkeep.  Output Y_b = out + b*out_bstride, Y_b[s*out_sf + k*out_sk] = fold of the entries i in [s*2^nlev, (s+1)*2^nlev) (zero beyond nfold) with challenges chal[b*chal_bstride + lev0 + l], l < nlev (0 <= nlev <= 4; nlev = 0 lifts). seg_fastest selects the thread mapping so that adjacent threads read adjacent memory.
template <typename TIN, bool EXTIN>
__device__ __forceinline__ void mle_fold_pass(const TIN *__restrict__ in, long long in_bstride, long long nfold,
                                              long long nkeep, long long sf, long long sk,
                                              const ext_t *__restrict__ chal, int chal_bstride, int lev0, int nlev,
                                              ext_t *__restrict__ out, long long out_bstride, long long out_sf,
                                              long long out_sk, int seg_fastest) {
    const int b = blockIdx.y;
    const long long seglen = 1LL << nlev;
    const long long nseg = (nfold + seglen - 1) >> nlev;
    const long long total = nseg * nkeep;
    const ext_t *rb = chal + (long long)b * chal_bstride + lev0;
    const TIN *X = in + (long long)b * in_bstride;
    ext_t *Y = out + (long long)b * out_bstride;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        long long s, k;
        if (seg_fastest) {
            s = idx % nseg;
            k = idx / nseg;
        } else {
            k = idx % nkeep;
            s = idx / nkeep;
        }
        const long long i0 = s << nlev;
        const TIN *base = X + k * sk;
        ext_t v[MLE_SEG];
#pragma unroll
        for (int e = 0; e < MLE_SEG; e++) {
            if (e < seglen) {
                const long long i = i0 + e;
                v[e] = (i < nfold) ? mle_load(base, i * sf) : ext_make(0ULL, 0ULL);
            }
        }
#pragma unroll
        for (int l = 0; l < MLE_SEG_LOG; l++) {
            if (l < nlev) {
                const ext_t r = rb[l];
                const int nout = (int)(seglen >> (l + 1));
#pragma unroll
                for (int j = 0; j < (MLE_SEG >> (l + 1)); j++) {
                    if (j < nout) {
                        if (l == 0 && !EXTIN)
                            v[j] = mle_base_fold(v[2 * j].c0, v[2 * j + 1].c0, r);
                        else
                            v[j] = ext_fold(v[2 * j], v[2 * j + 1], r);
                    }
                }
            }
        }
        Y[s * out_sf + k * out_sk] = v[0];
    }
}

#define MLE_FOLD_ARGS                                                                                   \
    long long in_bstride, long long nfold, long long nkeep, long long sf, long long sk,                 \
        const ext_t *__restrict__ chal, int chal_bstride, int lev0, int nlev, ext_t *__restrict__ out, \
        long long out_bstride, long long out_sf, long long out_sk, int seg_fastest
#define MLE_FOLD_PASS                                                                                   \
    in, in_bstride, nfold, nkeep, sf, sk, chal, chal_bstride, lev0, nlev, out, out_bstride, out_sf, out_sk, \
        seg_fastest

extern "C" __global__ void mle_fold_i8(const signed char *__restrict__ in, MLE_FOLD_ARGS) {
    mle_fold_pass<signed char, false>(MLE_FOLD_PASS);
}

extern "C" __global__ void mle_fold_u16(const unsigned short *__restrict__ in, MLE_FOLD_ARGS) {
    mle_fold_pass<unsigned short, false>(MLE_FOLD_PASS);
}

extern "C" __global__ void mle_fold_ext(const ext_t *__restrict__ in, MLE_FOLD_ARGS) {
    mle_fold_pass<ext_t, true>(MLE_FOLD_PASS);
}

// ----------------------------------------------------------------- SHA-256 (device, thread-local)
__constant__ unsigned int MLE_SHA_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

struct mle_sha_t {
    unsigned int h[8];
    unsigned char buf[64];
    int fill;
    unsigned long long len;
};

__device__ __forceinline__ unsigned int mle_rotr(unsigned int x, int n) { return (x >> n) | (x << (32 - n)); }

__device__ void mle_sha_block(mle_sha_t &c) {
    unsigned int w[64];
    for (int i = 0; i < 16; i++)
        w[i] = ((unsigned int)c.buf[4 * i] << 24) | ((unsigned int)c.buf[4 * i + 1] << 16) |
               ((unsigned int)c.buf[4 * i + 2] << 8) | (unsigned int)c.buf[4 * i + 3];
    for (int i = 16; i < 64; i++) {
        const unsigned int s0 = mle_rotr(w[i - 15], 7) ^ mle_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const unsigned int s1 = mle_rotr(w[i - 2], 17) ^ mle_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    unsigned int a = c.h[0], b = c.h[1], cc = c.h[2], d = c.h[3], e = c.h[4], f = c.h[5], g = c.h[6], h = c.h[7];
    for (int i = 0; i < 64; i++) {
        const unsigned int S1 = mle_rotr(e, 6) ^ mle_rotr(e, 11) ^ mle_rotr(e, 25);
        const unsigned int ch = (e & f) ^ (~e & g);
        const unsigned int t1 = h + S1 + ch + MLE_SHA_K[i] + w[i];
        const unsigned int S0 = mle_rotr(a, 2) ^ mle_rotr(a, 13) ^ mle_rotr(a, 22);
        const unsigned int mj = (a & b) ^ (a & cc) ^ (b & cc);
        const unsigned int t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1; d = cc; cc = b; b = a; a = t1 + t2;
    }
    c.h[0] += a; c.h[1] += b; c.h[2] += cc; c.h[3] += d; c.h[4] += e; c.h[5] += f; c.h[6] += g; c.h[7] += h;
}

__device__ void mle_sha_init(mle_sha_t &c) {
    c.h[0] = 0x6a09e667u; c.h[1] = 0xbb67ae85u; c.h[2] = 0x3c6ef372u; c.h[3] = 0xa54ff53au;
    c.h[4] = 0x510e527fu; c.h[5] = 0x9b05688cu; c.h[6] = 0x1f83d9abu; c.h[7] = 0x5be0cd19u;
    c.fill = 0;
    c.len = 0;
}

__device__ void mle_sha_byte(mle_sha_t &c, unsigned char x) {
    c.buf[c.fill++] = x;
    c.len++;
    if (c.fill == 64) {
        mle_sha_block(c);
        c.fill = 0;
    }
}

__device__ void mle_sha_update(mle_sha_t &c, const unsigned char *d, int n) {
    for (int i = 0; i < n; i++) mle_sha_byte(c, d[i]);
}

__device__ void mle_sha_u32le(mle_sha_t &c, unsigned int v) {
    for (int i = 0; i < 4; i++) mle_sha_byte(c, (unsigned char)(v >> (8 * i)));
}

__device__ void mle_sha_u64le(mle_sha_t &c, u64 v) {
    for (int i = 0; i < 8; i++) mle_sha_byte(c, (unsigned char)(v >> (8 * i)));
}

__device__ void mle_sha_ext(mle_sha_t &c, ext_t e) {
    mle_sha_u64le(c, e.c0);
    mle_sha_u64le(c, e.c1);
}

__device__ void mle_sha_final(mle_sha_t &c, unsigned char out[32]) {
    const unsigned long long bits = c.len * 8ULL;
    // padding: 0x80, zeros to 56 mod 64, 64-bit big-endian bit length (bypass len counter)
    c.buf[c.fill++] = 0x80;
    if (c.fill > 56) {
        while (c.fill < 64) c.buf[c.fill++] = 0;
        mle_sha_block(c);
        c.fill = 0;
    }
    while (c.fill < 56) c.buf[c.fill++] = 0;
    for (int i = 0; i < 8; i++) c.buf[56 + i] = (unsigned char)(bits >> (56 - 8 * i));
    mle_sha_block(c);
    for (int i = 0; i < 8; i++) {
        out[4 * i] = (unsigned char)(c.h[i] >> 24);
        out[4 * i + 1] = (unsigned char)(c.h[i] >> 16);
        out[4 * i + 2] = (unsigned char)(c.h[i] >> 8);
        out[4 * i + 3] = (unsigned char)c.h[i];
    }
}

__device__ __forceinline__ u64 mle_u64le(const unsigned char *p) {
    u64 v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

// golden round: state = SHA256(state || enc(s0) || enc(s1) || enc(s2)); r = decode(state)
__device__ ext_t mle_tr_round(unsigned char state[32], ext_t s0, ext_t s1, ext_t s2) {
    mle_sha_t c;
    mle_sha_init(c);
    mle_sha_update(c, state, 32);
    mle_sha_ext(c, s0);
    mle_sha_ext(c, s1);
    mle_sha_ext(c, s2);
    mle_sha_final(c, state);
    u64 r0 = mle_u64le(state), r1 = mle_u64le(state + 8);
    r0 -= GL_P & GL_MASK(r0 >= GL_P);
    r1 -= GL_P & GL_MASK(r1 >= GL_P);
    return ext_make(r0, r1);
}

// state_0 = SHA256("EDGE-CERT/sumfold/v1" || u32le(n) || enc(claim));
// bound:   SHA256("EDGE-CERT/sumfold-bind/v1" || state_0 || u32le(32) || ctx || u32le(len) || label)
__device__ void mle_tr_init_bound(unsigned char state[32], int n, ext_t claim, const unsigned char *ctx,
                                  const unsigned char *label, int label_len) {
    const char dom[] = "EDGE-CERT/sumfold/v1";
    const char bind[] = "EDGE-CERT/sumfold-bind/v1";
    mle_sha_t c;
    mle_sha_init(c);
    mle_sha_update(c, (const unsigned char *)dom, 20);
    mle_sha_u32le(c, (unsigned int)n);
    mle_sha_ext(c, claim);
    mle_sha_final(c, state);
    mle_sha_init(c);
    mle_sha_update(c, (const unsigned char *)bind, 25);
    mle_sha_update(c, state, 32);
    mle_sha_u32le(c, 32u);
    mle_sha_update(c, ctx, 32);
    mle_sha_u32le(c, (unsigned int)label_len);
    mle_sha_update(c, label, label_len);
    mle_sha_final(c, state);
}

// ----------------------------------------------------------------- batched product sumcheck
// Instance b = blockIdx.x:
//   level-0 tables  F + b*in_bstride, G + b*in_bstride  (read-only; entries >= m0 are zero)
//   scratch         S1f/S1g + b*s1_bstride (2^(n-1) ext), S2f/S2g + b*s2_bstride (2^(n-2) ext)
//                   level i >= 1 lives in S1 (i odd) or S2 (i even)
//   ctx             ctx + 32*b, label + b*label_bstride, label_len[b]
//   out             out + b*out_bstride: [0] claim, [1 + 3i + k] msgs, [1 + 3n + i] chal,
// [1 + 4n] final f, [2 + 4n] final g Launch: grid (batch), block (blockDim multiple of 32, <= 1024).
__device__ __forceinline__ ext_t mle_ld0(const ext_t *p, long long j, long long m0) {
    return (j < m0) ? p[j] : ext_make(0ULL, 0ULL);
}

extern "C" __global__ void mm_sumcheck(const ext_t *__restrict__ F, const ext_t *__restrict__ G,
                                       long long in_bstride, long long m0, int n, ext_t *S1f, ext_t *S1g,
                                       long long s1_bstride, ext_t *S2f, ext_t *S2g, long long s2_bstride,
                                       const unsigned char *__restrict__ ctx, const unsigned char *__restrict__ label,
                                       const int *__restrict__ label_len, int label_bstride,
                                       ext_t *partials, ext_t *out, long long out_bstride) {
    __shared__ ext_t sh_r;
    __shared__ unsigned char sh_state[32];
    const int b = blockIdx.x;
    const long long N = 1LL << n;
    const ext_t *f0 = F + (long long)b * in_bstride;
    const ext_t *g0 = G + (long long)b * in_bstride;
    ext_t *s1f = S1f + (long long)b * s1_bstride, *s1g = S1g + (long long)b * s1_bstride;
    ext_t *s2f = S2f + (long long)b * s2_bstride, *s2g = S2g + (long long)b * s2_bstride;
    ext_t *o = out + (long long)b * out_bstride;
    acc6_t A;

    // pass 0: evaluate level 0
    acc_zero(A);
    for (long long j = threadIdx.x; j < (N >> 1); j += blockDim.x) {
        eval_pair(A, mle_ld0(f0, 2 * j, m0), mle_ld0(f0, 2 * j + 1, m0), mle_ld0(g0, 2 * j, m0),
                  mle_ld0(g0, 2 * j + 1, m0));
    }
    block_reduce_store(A, partials);  // partials[3b..3b+2], written by thread 0
    if (threadIdx.x == 0) {
        const ext_t s0 = partials[3 * b], s1 = partials[3 * b + 1], s2 = partials[3 * b + 2];
        const ext_t claim = ext_add(s0, s1);
        int ll = label_len[b];
        if (ll > MLE_MAX_LABEL) ll = MLE_MAX_LABEL;
        mle_tr_init_bound(sh_state, n, claim, ctx + 32LL * b, label + (long long)b * label_bstride, ll);
        const ext_t r = mle_tr_round(sh_state, s0, s1, s2);
        o[0] = claim;
        o[1] = s0; o[2] = s1; o[3] = s2;
        o[1 + 3 * n] = r;
        sh_r = r;
    }
    __syncthreads();

    // fused passes i = 1 .. n-1: fold level i-1 with r_{i-1} -> level i, evaluate round i
    for (int i = 1; i < n; i++) {
        const ext_t r = sh_r;
        const long long nquads = N >> (i + 1);
        ext_t *df = (i & 1) ? s1f : s2f, *dg = (i & 1) ? s1g : s2g;
        const ext_t *sfp = (i == 1) ? f0 : ((i & 1) ? s2f : s1f);
        const ext_t *sgp = (i == 1) ? g0 : ((i & 1) ? s2g : s1g);
        const long long lim = (i == 1) ? m0 : (N >> (i - 1));
        acc_zero(A);
        for (long long k = threadIdx.x; k < nquads; k += blockDim.x) {
            const ext_t a0 = ext_fold(mle_ld0(sfp, 4 * k, lim), mle_ld0(sfp, 4 * k + 1, lim), r);
            const ext_t a1 = ext_fold(mle_ld0(sfp, 4 * k + 2, lim), mle_ld0(sfp, 4 * k + 3, lim), r);
            const ext_t b0 = ext_fold(mle_ld0(sgp, 4 * k, lim), mle_ld0(sgp, 4 * k + 1, lim), r);
            const ext_t b1 = ext_fold(mle_ld0(sgp, 4 * k + 2, lim), mle_ld0(sgp, 4 * k + 3, lim), r);
            df[2 * k] = a0;
            df[2 * k + 1] = a1;
            dg[2 * k] = b0;
            dg[2 * k + 1] = b1;
            eval_pair(A, a0, a1, b0, b1);
        }
        block_reduce_store(A, partials);
        if (threadIdx.x == 0) {
            const ext_t s0 = partials[3 * b], s1 = partials[3 * b + 1], s2 = partials[3 * b + 2];
            const ext_t rn = mle_tr_round(sh_state, s0, s1, s2);
            o[1 + 3 * i] = s0; o[2 + 3 * i] = s1; o[3 + 3 * i] = s2;
            o[1 + 3 * n + i] = rn;
            sh_r = rn;
        }
        __syncthreads();
    }

    // final fold of the 2-entry level n-1 table
    if (threadIdx.x == 0) {
        const ext_t r = sh_r;
        const int i = n - 1;
        const ext_t *sfp = (i == 0) ? f0 : ((i & 1) ? s1f : s2f);
        const ext_t *sgp = (i == 0) ? g0 : ((i & 1) ? s1g : s2g);
        const long long lim = (i == 0) ? m0 : 2;
        o[1 + 4 * n] = ext_fold(mle_ld0(sfp, 0, lim), mle_ld0(sfp, 1, lim), r);
        o[2 + 4 * n] = ext_fold(mle_ld0(sgp, 0, lim), mle_ld0(sgp, 1, lim), r);
    }
}
