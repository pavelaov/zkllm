/*
 * mle.c -- MLE partial evaluation (repeated folds), bound product sumcheck and eq-table verifier helpers for the TRANSPARENT_REFERENCE matmul prover.  See mle.h. Bit-exactness.  All values are canonical Goldilocks / Ext2 elements, so any correct computation of the same field element yields the same bits.  The fold code computes exactly the sequence of sumfold.fold() calls on the zero-padded table (the first fold of an integer table is written as base x ext, (a + r0*d, r1*d) with d = b - a, which equals emul(r, embed(d)) + embed(a) because the Karatsuba form gives (r0*d, r1*d) for c1 = 0). Fold engines (both are binary-counter streaming folds, equivalent to level-by-level folds): fast axis (sf == 1): one row of the fold axis is contiguous; chunks of 16 elements are folded completely in registers (4 levels), then pushed on a per-row stack of partial folds (entry at level l = fold of an aligned block of 2^l elements; two entries at the same level merge with r[l]).  Rows are independent (OpenMP over rows). general (slow) axis: the keep axis is tiled (64 columns); row pairs are folded into a 64-wide vector at level 1 and pushed on a stack of vectors.  For parallelism the fold axis is cut into aligned segments of 2^s rows (work item = segment x tile); the level-s segment results are folded with r[s..] in a second stage. Zero padding: when a stream ends, pending entries are folded with zero partners up to level nr (stack_finish); fold(v, 0, r) is computed like any other fold. Sumcheck engine 0: single-thread fused fold-and-evaluate (spec 3.1, in place, L = 1), lazy 128-bit round accumulators, transcript sf_transcript_init / sf_transcript_round from sumfold.c plus the optional binding hash of crypto/transcript.BoundSumcheckTranscript. Engine 1 calls the existing sf_prove() unchanged. Verifier helpers use a different algorithm (eq tables + lazily reduced integer x Ext multiply-accumulate) and share only gl_field.h with the prover.
*/
#include "mle.h"
#include "sha256.h"

#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if !defined(__GNUC__)
#error "mle.c needs gcc or clang (unsigned __int128); build with MinGW-w64 gcc (Makefile.mle)"
#endif

#ifndef MLE_BUILD_FLAGS
#define MLE_BUILD_FLAGS "unknown"
#endif

typedef unsigned __int128 u128;

static const ext_t EXT_ZERO = {0, 0};

#define CH_LOG 4                 /* fast-axis chunk: 16 elements folded in registers */
#define CH (1 << CH_LOG)
#define TILE_W 64                /* slow-axis tile width (keep axis) */
#define MLE_PAR_MIN 32768        /* fewer elements than this: always serial */
#define MLE_MAX_THREADS 256

const char *mle_build_info(void) { return MLE_BUILD_FLAGS; }

/* ------------------------------------------------------------ scratch accounting */
static int64_t g_scr_cur = 0, g_scr_peak = 0;

static void *scr_alloc(size_t bytes) {
    void *p = malloc(bytes ? bytes : 1);
    if (p) {
        int64_t cur = __atomic_add_fetch(&g_scr_cur, (int64_t)bytes, __ATOMIC_RELAXED);
        int64_t pk = __atomic_load_n(&g_scr_peak, __ATOMIC_RELAXED);
        while (cur > pk && !__atomic_compare_exchange_n(&g_scr_peak, &pk, cur, 1, __ATOMIC_RELAXED,
                                                        __ATOMIC_RELAXED)) {
        }
    }
    return p;
}

static void scr_free(void *p, size_t bytes) {
    if (p) {
        free(p);
        __atomic_sub_fetch(&g_scr_cur, (int64_t)bytes, __ATOMIC_RELAXED);
    }
}

int64_t mle_scratch_peak(void) { return __atomic_load_n(&g_scr_peak, __ATOMIC_RELAXED); }
int64_t mle_scratch_current(void) { return __atomic_load_n(&g_scr_cur, __ATOMIC_RELAXED); }
void mle_scratch_reset(void) { __atomic_store_n(&g_scr_peak, mle_scratch_current(), __ATOMIC_RELAXED); }

/* ------------------------------------------------------------------------ lift */
/* v mod p for any int64 v (field.lift_signed): v >= 0 is < 2^63 < p; v < 0 maps to p + v. */
GL_INLINE uint64_t lift_i64(int64_t v) { return (uint64_t)v + (GL_P & GL_MASK(v < 0)); }

static void load_lift_run(const void *M, int dt, int64_t off, int64_t stride, int64_t cnt, uint64_t *dst) {
    int64_t e;
#define LLR_CASE(CODE, TYPE)                                                                   \
    case CODE: {                                                                               \
        const TYPE *p = (const TYPE *)M + off;                                                 \
        if (stride == 1)                                                                       \
            for (e = 0; e < cnt; e++) dst[e] = lift_i64((int64_t)p[e]);                        \
        else                                                                                   \
            for (e = 0; e < cnt; e++) dst[e] = lift_i64((int64_t)p[e * stride]);               \
    } break;
    switch (dt) {
        LLR_CASE(MLE_I8, int8_t)
        LLR_CASE(MLE_U8, uint8_t)
        LLR_CASE(MLE_I16, int16_t)
        LLR_CASE(MLE_U16, uint16_t)
        LLR_CASE(MLE_I32, int32_t)
        LLR_CASE(MLE_I64, int64_t)
        default:
            for (e = 0; e < cnt; e++) dst[e] = 0;
            break;
    }
#undef LLR_CASE
}

int mle_lift(const void *M, int dtype, int64_t n, int64_t stride, uint64_t *dst) {
    if (!M || !dst || n < 0 || dtype < MLE_I8 || dtype > MLE_I64) return -1;
    load_lift_run(M, dtype, 0, stride, n, dst);
    return 0;
}

/* ------------------------------------------------------------------------ folds */
/* fold(embed(a), embed(b), r) for base-field a, b: (a + r0 d, r1 d), d = b - a. */
GL_INLINE ext_t base_fold(uint64_t a, uint64_t b, ext_t r) {
    const uint64_t d = gl_sub(b, a);
    ext_t o;
    o.c0 = gl_add(a, gl_mul(r.c0, d));
    o.c1 = gl_mul(r.c1, d);
    return o;
}

/* Fold 2^c lifted base values (c <= CH_LOG) completely with r[0..c-1]. */
GL_INLINE ext_t fold_chunk(const uint64_t *v, int c, const ext_t *r) {
    ext_t w[CH / 2];
    int h, j, l;
    if (c == 0) {
        ext_t o;
        o.c0 = v[0];
        o.c1 = 0;
        return o;
    }
    h = 1 << (c - 1);
    for (j = 0; j < h; j++) w[j] = base_fold(v[2 * j], v[2 * j + 1], r[0]);
    for (l = 1; l < c; l++) {
        h >>= 1;
        for (j = 0; j < h; j++) w[j] = ext_fold(w[2 * j], w[2 * j + 1], r[l]);
    }
    return w[0];
}

/*
 * Finish a binary-counter fold.  st[l] (mask bit l) is the fold of a complete aligned block of 2^l entries, for levels l >= lo; the blocks are the binary digits of the number of level-lo blocks seen.  The rest of the length-2^nr table is zero.  Walking up the levels, a pending entry is a left child whose right sibling is the carry (or zero); without a pending entry the carry is a left child with a zero right sibling.
*/
static ext_t stack_finish(const ext_t *st, uint64_t mask, int lo, int nr, const ext_t *r) {
    ext_t carry = EXT_ZERO;
    int have = 0, l;
    if ((mask >> nr) & 1) return st[nr];
    for (l = lo; l < nr; l++) {
        if ((mask >> l) & 1) {
            carry = ext_fold(st[l], have ? carry : EXT_ZERO, r[l]);
            have = 1;
        } else if (have) {
            carry = ext_fold(carry, EXT_ZERO, r[l]);
        }
    }
    return carry; /* zero if the stream was empty */
}

/* Fast axis: m contiguous integers starting at element offset off. */
static ext_t fold_row(const void *M, int dt, int64_t off, int64_t m, const ext_t *r, int nr) {
    ext_t st[MLE_MAX_VARS + 1];
    uint64_t v[CH];
    uint64_t mask = 0;
    const int c = nr < CH_LOG ? nr : CH_LOG;
    const int64_t ch = (int64_t)1 << c;
    int64_t base;
    for (base = 0; base < m; base += ch) {
        const int64_t lim = (m - base) < ch ? (m - base) : ch;
        ext_t u;
        int l;
        load_lift_run(M, dt, off + base, 1, lim, v);
        if (lim < ch) memset(v + lim, 0, (size_t)(ch - lim) * sizeof(uint64_t));
        u = (c == CH_LOG) ? fold_chunk(v, CH_LOG, r) : fold_chunk(v, c, r);
        l = c;
        while ((mask >> l) & 1) {
            u = ext_fold(st[l], u, r[l]);
            mask &= ~((uint64_t)1 << l);
            l++;
        }
        st[l] = u;
        mask |= (uint64_t)1 << l;
    }
    return stack_finish(st, mask, c, nr, r);
}

static int fold_fast(const void *M, int dt, int64_t nfold, int64_t nkeep, int64_t sk, const ext_t *r,
                     int nr, ext_t *out, int threads) {
    int64_t k;
    if (threads <= 1) {
        for (k = 0; k < nkeep; k++) out[k] = fold_row(M, dt, k * sk, nfold, r, nr);
        return 1;
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic, 4)
#endif
    for (k = 0; k < nkeep; k++) out[k] = fold_row(M, dt, k * sk, nfold, r, nr);
    return threads;
}

typedef struct {
    ext_t *st, *u;
    uint64_t *v0, *v1;
    int64_t bw;
    int levels;
} tile_scr_t;

static int tile_scr_init(tile_scr_t *S, int64_t bw, int levels) {
    S->bw = bw;
    S->levels = levels;
    S->st = (ext_t *)scr_alloc((size_t)(levels + 1) * (size_t)bw * sizeof(ext_t));
    S->u = (ext_t *)scr_alloc((size_t)bw * sizeof(ext_t));
    S->v0 = (uint64_t *)scr_alloc((size_t)bw * sizeof(uint64_t));
    S->v1 = (uint64_t *)scr_alloc((size_t)bw * sizeof(uint64_t));
    return S->st && S->u && S->v0 && S->v1;
}

static void tile_scr_free(tile_scr_t *S) {
    scr_free(S->st, (size_t)(S->levels + 1) * (size_t)S->bw * sizeof(ext_t));
    scr_free(S->u, (size_t)S->bw * sizeof(ext_t));
    scr_free(S->v0, (size_t)S->bw * sizeof(uint64_t));
    scr_free(S->v1, (size_t)S->bw * sizeof(uint64_t));
}

/*
 * Fold rows [i0, i1) of the tile of columns [k0, k0 + bw) with r[0..s-1] -> level-s vector. i0 is a multiple of 2^s and i1 - i0 <= 2^s; rows >= i1 are zero.  Element (i, k) at M[i*sf + k*sk]; dt MLE_EXT means ext_t input (first fold is then ext x ext).
*/
static void fold_tile(const void *M, int dt, int64_t sf, int64_t sk, int64_t i0, int64_t i1, int s, int64_t k0,
                      int64_t bw, const ext_t *r, tile_scr_t *S, ext_t *out) {
    uint64_t mask = 0;
    int64_t i, b;
    int l;
    const int64_t BW = S->bw;
    if (s == 0) { /* a single row, no fold */
        if (dt == MLE_EXT) {
            const ext_t *X = (const ext_t *)M;
            for (b = 0; b < bw; b++) out[b] = X[i0 * sf + (k0 + b) * sk];
        } else {
            load_lift_run(M, dt, i0 * sf + k0 * sk, sk, bw, S->v0);
            for (b = 0; b < bw; b++) {
                out[b].c0 = S->v0[b];
                out[b].c1 = 0;
            }
        }
        return;
    }
    for (i = i0; i < i1; i += 2) {
        const int has1 = (i + 1) < i1;
        ext_t *u = S->u;
        if (dt == MLE_EXT) {
            const ext_t *X = (const ext_t *)M;
            for (b = 0; b < bw; b++) {
                const ext_t a = X[i * sf + (k0 + b) * sk];
                const ext_t c = has1 ? X[(i + 1) * sf + (k0 + b) * sk] : EXT_ZERO;
                u[b] = ext_fold(a, c, r[0]);
            }
        } else {
            load_lift_run(M, dt, i * sf + k0 * sk, sk, bw, S->v0);
            if (has1)
                load_lift_run(M, dt, (i + 1) * sf + k0 * sk, sk, bw, S->v1);
            else
                memset(S->v1, 0, (size_t)bw * sizeof(uint64_t));
            for (b = 0; b < bw; b++) u[b] = base_fold(S->v0[b], S->v1[b], r[0]);
        }
        l = 1;
        while ((mask >> l) & 1) {
            const ext_t *t = S->st + (int64_t)l * BW;
            for (b = 0; b < bw; b++) u[b] = ext_fold(t[b], u[b], r[l]);
            mask &= ~((uint64_t)1 << l);
            l++;
        }
        memcpy(S->st + (int64_t)l * BW, u, (size_t)bw * sizeof(ext_t));
        mask |= (uint64_t)1 << l;
    }
    if ((mask >> s) & 1) {
        memcpy(out, S->st + (int64_t)s * BW, (size_t)bw * sizeof(ext_t));
        return;
    }
    {
        int have = 0;
        for (l = 1; l < s; l++) {
            if ((mask >> l) & 1) {
                const ext_t *t = S->st + (int64_t)l * BW;
                for (b = 0; b < bw; b++) out[b] = ext_fold(t[b], have ? out[b] : EXT_ZERO, r[l]);
                have = 1;
            } else if (have) {
                for (b = 0; b < bw; b++) out[b] = ext_fold(out[b], EXT_ZERO, r[l]);
            }
        }
        if (!have) memset(out, 0, (size_t)bw * sizeof(ext_t));
    }
}

/* one work item of the slow-axis engine */
static void slow_item(const void *M, int dt, int64_t nfold, int64_t nkeep, int64_t sf, int64_t sk, const ext_t *r,
                      int s, int64_t tiles, int64_t w, ext_t *dst, tile_scr_t *S) {
    const int64_t seg = w / tiles, tile = w % tiles;
    const int64_t k0 = tile * TILE_W;
    const int64_t bw = (nkeep - k0) < TILE_W ? (nkeep - k0) : TILE_W;
    const int64_t i0 = seg << s;
    const int64_t i1 = (i0 + ((int64_t)1 << s)) < nfold ? (i0 + ((int64_t)1 << s)) : nfold;
    fold_tile(M, dt, sf, sk, i0, i1, s, k0, bw, r, S, dst + seg * nkeep + k0);
}

static int run_items(const void *M, int dt, int64_t nfold, int64_t nkeep, int64_t sf, int64_t sk, const ext_t *r,
                     int s, int64_t items, int64_t tiles, ext_t *dst, int threads) {
    int err = 0;
    if (threads <= 1 || items <= 1) {
        tile_scr_t S;
        int64_t w;
        if (!tile_scr_init(&S, TILE_W, s)) err = 1;
        else
            for (w = 0; w < items; w++) slow_item(M, dt, nfold, nkeep, sf, sk, r, s, tiles, w, dst, &S);
        tile_scr_free(&S);
        return err ? -7 : 1;
    }
#ifdef _OPENMP
#pragma omp parallel num_threads(threads) reduction(| : err)
#endif
    {
        tile_scr_t S;
        int64_t w;
        if (!tile_scr_init(&S, TILE_W, s)) err = 1;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (w = 0; w < items; w++)
            if (!err) slow_item(M, dt, nfold, nkeep, sf, sk, r, s, tiles, w, dst, &S);
        tile_scr_free(&S);
    }
    return err ? -7 : threads;
}

static int fold_slow(const void *M, int dt, int64_t nfold, int64_t nkeep, int64_t sf, int64_t sk, const ext_t *r,
                     int nr, ext_t *out, int threads) {
    const int64_t tiles = (nkeep + TILE_W - 1) / TILE_W;
    int s = nr, used;
    int64_t nseg;
    ext_t *part = out;
    size_t part_bytes = 0;
    if (threads > 1) { /* cut the fold axis until there are >= 4 work items per thread */
        while (s > CH_LOG && tiles * ((nfold + ((int64_t)1 << s) - 1) >> s) < 4 * (int64_t)threads) s--;
    }
    nseg = (nfold + ((int64_t)1 << s) - 1) >> s;
    if (nseg > 1) {
        part_bytes = (size_t)nseg * (size_t)nkeep * sizeof(ext_t);
        part = (ext_t *)scr_alloc(part_bytes);
        if (!part) return -6;
    }
    used = run_items(M, dt, nfold, nkeep, sf, sk, r, s, nseg * tiles, tiles, part, threads);
    if (used > 0 && nseg > 1) { /* stage 2: fold the level-s segment results with r[s..nr-1] */
        int u2 = run_items(part, MLE_EXT, nseg, nkeep, nkeep, 1, r + s, nr - s, tiles, tiles, out,
                           tiles >= 2 ? threads : 1);
        if (u2 < 0) used = u2;
    }
    if (nseg > 1) scr_free(part, part_bytes);
    return used;
}

int mle_fold_axis(const void *M, int dtype, int64_t nfold, int64_t nkeep, int64_t sf, int64_t sk, const ext_t *r,
                  int nr, ext_t *out, int threads) {
    if (!M || !out || nfold < 1 || nkeep < 1 || nr < 0 || nr > MLE_MAX_VARS) return -1;
    if (nr > 0 && !r) return -1;
    if (nfold > ((int64_t)1 << nr)) return -2;
    if (dtype < MLE_I8 || dtype > MLE_EXT) return -3;
    if (threads < 1) threads = 1;
    if (threads > MLE_MAX_THREADS) threads = MLE_MAX_THREADS;
    if (nfold * nkeep < MLE_PAR_MIN) threads = 1;
    if (sf == 1 && dtype != MLE_EXT && nfold > 1)
        return fold_fast(M, dtype, nfold, nkeep, sk, r, nr, out, threads);
    return fold_slow(M, dtype, nfold, nkeep, sf, sk, r, nr, out, threads);
}

/* ------------------------------------------------------------------ sumcheck */
typedef struct {
    acc128_t a[6];
} macc6_t;

GL_INLINE void m_eval_pair(macc6_t *A, ext_t f0, ext_t f1, ext_t g0, ext_t g1) {
    const ext_t p0 = ext_mul(f0, g0);
    const ext_t p1 = ext_mul(f1, g1);
    const ext_t p2 = ext_mul(ext_sub(ext_dbl(f1), f0), ext_sub(ext_dbl(g1), g0));
    acc_add(&A->a[0], p0.c0);
    acc_add(&A->a[1], p0.c1);
    acc_add(&A->a[2], p1.c0);
    acc_add(&A->a[3], p1.c1);
    acc_add(&A->a[4], p2.c0);
    acc_add(&A->a[5], p2.c1);
}

static void macc_finish(const macc6_t *A, ext_t out[3]) {
    int k;
    for (k = 0; k < 3; k++) {
        out[k].c0 = acc_reduce(A->a[2 * k]);
        out[k].c1 = acc_reduce(A->a[2 * k + 1]);
    }
}

static const char BIND_DST[] = "EDGE-CERT/sumfold-bind/v1"; /* 25 bytes, no NUL */

static void put_u32le(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static void bind_state(sf_transcript_t *tr, const uint8_t *ctx, int ctx_len, const uint8_t *label, int label_len) {
    sha256_ctx c;
    uint8_t le[4];
    sha256_init(&c);
    sha256_update(&c, BIND_DST, 25);
    sha256_update(&c, tr->state, 32);
    put_u32le(le, (uint32_t)ctx_len);
    sha256_update(&c, le, 4);
    if (ctx_len > 0) sha256_update(&c, ctx, (size_t)ctx_len);
    put_u32le(le, (uint32_t)label_len);
    sha256_update(&c, le, 4);
    if (label_len > 0) sha256_update(&c, label, (size_t)label_len);
    sha256_final(&c, tr->state);
}

static void sc_engine0(ext_t *f, ext_t *g, int n, int bind, const uint8_t *ctx, int ctx_len, const uint8_t *label,
                       int label_len, mle_sc_proof_t *out) {
    const int64_t N = (int64_t)1 << n;
    sf_transcript_t tr;
    macc6_t A;
    ext_t s[3], r;
    int64_t j, k, Kq;
    int i;
    memset(&A, 0, sizeof A);
    for (j = 0; j < N; j += 2) m_eval_pair(&A, f[j], f[j + 1], g[j], g[j + 1]);
    macc_finish(&A, s);
    out->n = n;
    out->claim = ext_add(s[0], s[1]); /* == inner(f, g) */
    sf_transcript_init(&tr, n, out->claim);
    if (bind) bind_state(&tr, ctx, ctx_len, label, label_len);
    r = sf_transcript_round(&tr, s[0], s[1], s[2]);
    out->msgs[0][0] = s[0];
    out->msgs[0][1] = s[1];
    out->msgs[0][2] = s[2];
    out->chal[0] = r;
    for (i = 1; i < n; i++) {
        Kq = (N >> (i - 1)) >> 2;
        memset(&A, 0, sizeof A);
        for (k = 0; k < Kq; k++) {
            const ext_t a0 = ext_fold(f[4 * k], f[4 * k + 1], r), a1 = ext_fold(f[4 * k + 2], f[4 * k + 3], r);
            const ext_t b0 = ext_fold(g[4 * k], g[4 * k + 1], r), b1 = ext_fold(g[4 * k + 2], g[4 * k + 3], r);
            f[2 * k] = a0;
            f[2 * k + 1] = a1;
            g[2 * k] = b0;
            g[2 * k + 1] = b1;
            m_eval_pair(&A, a0, a1, b0, b1);
        }
        macc_finish(&A, s);
        r = sf_transcript_round(&tr, s[0], s[1], s[2]);
        out->msgs[i][0] = s[0];
        out->msgs[i][1] = s[1];
        out->msgs[i][2] = s[2];
        out->chal[i] = r;
    }
    out->final_f = ext_fold(f[0], f[1], r);
    out->final_g = ext_fold(g[0], g[1], r);
}

int mle_sumcheck(ext_t *f, ext_t *g, int n, const uint8_t *ctx, int ctx_len, const uint8_t *label, int label_len,
                 int bind, int engine, int threads, mle_sc_proof_t *out) {
    if (!f || !g || !out || n < 1 || n >= MLE_MAX_VARS) return -1;
    if (bind && ((ctx_len > 0 && !ctx) || (label_len > 0 && !label) || ctx_len < 0 || label_len < 0)) return -1;
    memset(out, 0, sizeof *out);
    if (engine == 0) {
        sc_engine0(f, g, n, bind, ctx, ctx_len, label, label_len, out);
        return 0;
    }
    if (engine == 1) {
        sf_config_t cfg;
        sf_result_t *res;
        int rc, i;
        if (bind) return -4; /* sf_prove implements the golden (unbound) transcript only */
        if (n >= SF_MAX_LOGN) return -1;
        memset(&cfg, 0, sizeof cfg);
        cfg.variant = SF_FUSED;
        cfg.n = n;
        cfg.threads = threads < 1 ? 1 : threads;
        res = (sf_result_t *)malloc(sizeof(sf_result_t));
        if (!res) return -6;
        rc = sf_prove(&cfg, f, g, NULL, NULL, res);
        if (rc == 0) {
            out->n = n;
            out->claim = res->claim;
            for (i = 0; i < n; i++) {
                out->msgs[i][0] = res->msgs[i][0];
                out->msgs[i][1] = res->msgs[i][1];
                out->msgs[i][2] = res->msgs[i][2];
                out->chal[i] = res->chal[i];
            }
            out->final_f = res->final_f;
            out->final_g = res->final_g;
        }
        free(res);
        return rc;
    }
    return -5;
}

/* ------------------------------------------------------------------ matmul */
int mm_prove(const void *A, int dtA, int64_t T, int64_t K, int64_t sAt, int64_t sAk, const void *W, int dtW,
             int64_t N, int64_t sWk, int64_t sWn, const ext_t *r1, int t, const ext_t *r2, int nb,
             const uint8_t *ctx, int ctx_len, const uint8_t *label, int label_len, int threads,
             mle_sc_proof_t *out, mm_times_t *tm, ext_t *a_out, ext_t *w_out) {
    int kappa = 0, rc, used_a, used_w;
    int64_t NK, t0, t1, t2, t3;
    size_t tb;
    ext_t *f, *g;
    const double sec = sf_tick_seconds();
    if (!A || !W || !out || T < 1 || K < 2 || N < 1) return -1;
    if (dtA == MLE_EXT || dtW == MLE_EXT) return -3;
    while (((int64_t)1 << kappa) < K) kappa++;
    if (kappa >= MLE_MAX_VARS) return -1;
    NK = (int64_t)1 << kappa;
    tb = (size_t)NK * sizeof(ext_t);
    f = (ext_t *)scr_alloc(tb);
    g = (ext_t *)scr_alloc(tb);
    if (!f || !g) {
        scr_free(f, tb);
        scr_free(g, tb);
        return -6;
    }
    t0 = sf_ticks();
    used_a = mle_fold_axis(A, dtA, T, K, sAt, sAk, r1, t, f, threads);
    t1 = sf_ticks();
    used_w = mle_fold_axis(W, dtW, N, K, sWn, sWk, r2, nb, g, threads);
    t2 = sf_ticks();
    if (used_a < 0 || used_w < 0) {
        scr_free(f, tb);
        scr_free(g, tb);
        return used_a < 0 ? used_a : used_w;
    }
    if (a_out) memcpy(a_out, f, (size_t)K * sizeof(ext_t));
    if (w_out) memcpy(w_out, g, (size_t)K * sizeof(ext_t));
    memset(f + K, 0, (size_t)(NK - K) * sizeof(ext_t));
    memset(g + K, 0, (size_t)(NK - K) * sizeof(ext_t));
    rc = mle_sumcheck(f, g, kappa, ctx, ctx_len, label, label_len, 1, 0, 1, out);
    t3 = sf_ticks();
    if (tm) {
        tm->fold_a_s = (double)(t1 - t0) * sec;
        tm->fold_w_s = (double)(t2 - t1) * sec;
        tm->sumcheck_s = (double)(t3 - t2) * sec;
        tm->total_s = (double)(t3 - t0) * sec;
        tm->table_bytes = (int64_t)(2 * tb);
        tm->kappa = kappa;
        tm->threads_used = used_a > used_w ? used_a : used_w;
    }
    scr_free(f, tb);
    scr_free(g, tb);
    return rc;
}

/* ------------------------------------------------------------------ verifier side */
int mlev_eq_table(const ext_t *r, int nr, ext_t *out) {
    int64_t size = 1, i;
    int t;
    if (!out || nr < 0 || nr > MLE_MAX_VARS || (nr > 0 && !r)) return -1;
    out[0].c0 = 1;
    out[0].c1 = 0;
    for (t = 0; t < nr; t++) {
        for (i = 0; i < size; i++) {
            const ext_t hi = ext_mul(out[i], r[t]);
            out[i + size] = hi;
            out[i] = ext_sub(out[i], hi); /* e * (1 - r_t) */
        }
        size <<= 1;
    }
    return 0;
}

#define LIN_CHUNK 4096

GL_INLINE uint64_t red128(u128 x) { return gl_reduce128((uint64_t)x, (uint64_t)(x >> 64)); }

/*
 * sum_{i < ni} M[off + i*si] * e[i] for an integer row.  Signed entries are offset by OFF so the products (m + OFF) * e are non-negative; exact 128-bit sums over chunks of LIN_CHUNK entries (< 2^32 * 2^64 * 2^12 = 2^108), then  sum (m + OFF) e - OFF * sum e.
*/
static ext_t lin_row(const void *M, int dt, int64_t off, int64_t ni, int64_t si, const ext_t *e) {
    ext_t acc = EXT_ZERO;
    int64_t c0;
    for (c0 = 0; c0 < ni; c0 += LIN_CHUNK) {
        const int64_t c1 = (c0 + LIN_CHUNK) < ni ? (c0 + LIN_CHUNK) : ni;
        u128 a0 = 0, a1 = 0, s0 = 0, s1 = 0;
        uint64_t OFF = 0;
        ext_t x;
        int64_t i;
#define LIN_CASE(CODE, TYPE, OFFV)                                                            \
    case CODE: {                                                                              \
        const TYPE *p = (const TYPE *)M + off;                                                \
        OFF = (OFFV);                                                                         \
        for (i = c0; i < c1; i++) {                                                           \
            const uint64_t m = (uint64_t)((int64_t)p[i * si] + (int64_t)(OFFV));              \
            a0 += (u128)m * e[i].c0;                                                          \
            a1 += (u128)m * e[i].c1;                                                          \
            if (OFFV) {                                                                       \
                s0 += e[i].c0;                                                                \
                s1 += e[i].c1;                                                                \
            }                                                                                 \
        }                                                                                     \
    } break;
        switch (dt) {
            LIN_CASE(MLE_I8, int8_t, 128)
            LIN_CASE(MLE_U8, uint8_t, 0)
            LIN_CASE(MLE_I16, int16_t, 32768)
            LIN_CASE(MLE_U16, uint16_t, 0)
            LIN_CASE(MLE_I32, int32_t, 2147483648LL)
            case MLE_I64: { /* rare: full field multiply per entry */
                const int64_t *p = (const int64_t *)M + off;
                uint64_t x0 = 0, x1 = 0;
                for (i = c0; i < c1; i++) {
                    const uint64_t m = lift_i64(p[i * si]);
                    x0 = gl_add(x0, gl_mul(m, e[i].c0));
                    x1 = gl_add(x1, gl_mul(m, e[i].c1));
                }
                a0 = x0;
                a1 = x1;
            } break;
            default:
                break;
        }
#undef LIN_CASE
        x.c0 = red128(a0);
        x.c1 = red128(a1);
        if (OFF) {
            x.c0 = gl_sub(x.c0, gl_mul(OFF, red128(s0)));
            x.c1 = gl_sub(x.c1, gl_mul(OFF, red128(s1)));
        }
        acc = ext_add(acc, x);
    }
    return acc;
}

int mlev_bilinear(const void *M, int dtype, int64_t R, int64_t C, int64_t sR, int64_t sC, const ext_t *eqR,
                  const ext_t *eqC, int threads, ext_t *out) {
    int64_t no, ni, so, si;
    const ext_t *eqO, *eqI;
    ext_t part[MLE_MAX_THREADS];
    int nth = threads < 1 ? 1 : (threads > MLE_MAX_THREADS ? MLE_MAX_THREADS : threads), t;
    ext_t tot = EXT_ZERO;
    if (!M || !eqR || !eqC || !out || R < 1 || C < 1 || dtype < MLE_I8 || dtype > MLE_I64) return -1;
    if (sC == 1 || sR != 1) {
        no = R; ni = C; so = sR; si = sC; eqO = eqR; eqI = eqC;
    } else {
        no = C; ni = R; so = sC; si = sR; eqO = eqC; eqI = eqR;
    }
    if (R * C < MLE_PAR_MIN) nth = 1;
    for (t = 0; t < nth; t++) part[t] = EXT_ZERO;
    if (nth == 1) {
        int64_t o;
        for (o = 0; o < no; o++) part[0] = ext_add(part[0], ext_mul(eqO[o], lin_row(M, dtype, o * so, ni, si, eqI)));
    } else {
#ifdef _OPENMP
#pragma omp parallel num_threads(nth)
        {
            const int tid = omp_get_thread_num();
            ext_t acc = EXT_ZERO;
            int64_t o;
#pragma omp for schedule(static)
            for (o = 0; o < no; o++) acc = ext_add(acc, ext_mul(eqO[o], lin_row(M, dtype, o * so, ni, si, eqI)));
            part[tid] = acc;
        }
#else
        int64_t o;
        for (o = 0; o < no; o++) part[0] = ext_add(part[0], ext_mul(eqO[o], lin_row(M, dtype, o * so, ni, si, eqI)));
#endif
    }
    for (t = 0; t < nth; t++) tot = ext_add(tot, part[t]);
    *out = tot;
    return 0;
}
