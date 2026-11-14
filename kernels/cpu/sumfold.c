/*
 * sumfold.c -- CPU SumFold prover engines (fused / unfused), SHA-256 transcript. Bit-exact with src/edgecert/sumfold.py::prove() (see spec/sumfold_kernel.md). Claim.  prove() computes H = inner(f, g) with a separate pass.  Here H is taken as s0 + s1 of round 0 (sum over even indices plus sum over odd indices), which is the same field element exactly, so the extra pass is not needed.  The transcript is initialised with H right after pass 0 and before r_0 is derived, as in prove(). Parallelism.  One OpenMP parallel region per engine run; per-thread partial round sums (cache-line padded) are reduced by the master thread, which also runs the transcript; barriers separate passes.  Field addition is exact, so the reduction order does not change any output bit.  The pass barriers are a sense-reversing spin barrier (sf_barrier_*), not '#pragma omp barrier': the MinGW libgomp barrier costs ~0.1 ms per crossing on this host, which would dominate small n. Fused engine (spec 3.1 + 3.2): the tables are split into L lanes (L a power of two); lane l owns the contiguous block [l*B, (l+1)*B), B = N/L.  Each lane is folded in place inside its own block (write index 2k, 2k+1 trails read index 4k..4k+3), so no ping-pong buffer is needed and threads never touch another lane's block.  After R = n - log2(L) rounds each lane holds one (f, g) pair; the residues are gathered (in place, ascending) and the last log2(L) rounds run single-threaded ("host tail"). Unfused engine (ablation baseline): per round an evaluate pass over the current table, the transcript, then an out-of-place fold pass into a ping-pong buffer.
*/
#include "sumfold.h"
#include "sha256.h"

#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_thread_num(void) { return 0; }
static int omp_get_num_threads(void) { return 1; }
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sched.h>
#include <time.h>
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define SF_ALIGN64 __declspec(align(64))
#include <intrin.h>
#define SF_PAUSE() _mm_pause()
#define SF_FETCH_ADD(p, v) _InterlockedExchangeAdd((volatile long *)(p), (long)(v))
#define SF_LOAD_ACQ(p) (*(volatile long *)(p))            /* x64 + /volatile:ms: acquire */
#define SF_STORE_REL(p, v) (*(volatile long *)(p) = (long)(v)) /* x64 + /volatile:ms: release */
#else
#define SF_ALIGN64 __attribute__((aligned(64)))
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define SF_PAUSE() _mm_pause()
#else
#define SF_PAUSE() ((void)0)
#endif
#define SF_FETCH_ADD(p, v) __atomic_fetch_add((p), (v), __ATOMIC_ACQ_REL)
#define SF_LOAD_ACQ(p) __atomic_load_n((p), __ATOMIC_ACQUIRE)
#define SF_STORE_REL(p, v) __atomic_store_n((p), (v), __ATOMIC_RELEASE)
#endif

/* ------------------------------------------------------- spin barrier */
typedef struct {
    volatile long count;
    char pad0[60];
    volatile long sense; /* separate cache line from count */
    char pad1[60];
} sf_barrier_t;

static SF_ALIGN64 sf_barrier_t g_barrier; /* engines are not re-entrant (see g_partial) */

static void sf_barrier_reset(sf_barrier_t *b) {
    b->count = 0;
    b->sense = 0;
}

static void sf_yield(void) {
#ifdef _WIN32
    SwitchToThread();
#else
    sched_yield();
#endif
}

/* Sense-reversing centralized barrier for a team of nth threads. */
static void sf_barrier_wait(sf_barrier_t *b, int nth, long *local_sense) {
    const long s = 1 - *local_sense;
    *local_sense = s;
    if (nth <= 1) return;
    if (SF_FETCH_ADD(&b->count, 1) == nth - 1) {
        b->count = 0;
        SF_STORE_REL(&b->sense, s);
    } else {
        /* short spin (a pause is ~100+ cycles on recent Intel cores), then yield so a
         * descheduled team member can run when all logical CPUs are occupied */
        unsigned spins = 0;
        while (SF_LOAD_ACQ(&b->sense) != s) {
            if (spins < 256u) { SF_PAUSE(); spins++; }
            else sf_yield();
        }
    }
}

/* ------------------------------------------------------------------ timer */
int64_t sf_ticks(void) {
#ifdef _WIN32
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (int64_t)t.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
#endif
}

double sf_tick_seconds(void) {
#ifdef _WIN32
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    return 1.0 / (double)f.QuadPart;
#else
    return 1e-9;
#endif
}

/* ------------------------------------------------------------- transcript */
static const char SF_DOMAIN[] = "EDGE-CERT/sumfold/v1"; /* 20 bytes, no NUL */

static void put_u64le(uint8_t *p, uint64_t v) {
    int i;
    for (i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

static uint64_t get_u64le(const uint8_t *p) {
    uint64_t v = 0;
    int i;
    for (i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

static void put_ext(uint8_t *p, ext_t e) {
    put_u64le(p, e.c0);
    put_u64le(p + 8, e.c1);
}

void sf_transcript_init(sf_transcript_t *t, int n, ext_t claim) {
    uint8_t buf[20 + 4 + 16];
    uint32_t un = (uint32_t)n;
    memcpy(buf, SF_DOMAIN, 20);
    buf[20] = (uint8_t)un;
    buf[21] = (uint8_t)(un >> 8);
    buf[22] = (uint8_t)(un >> 16);
    buf[23] = (uint8_t)(un >> 24);
    put_ext(buf + 24, claim);
    sha256(buf, sizeof buf, t->state);
}

ext_t sf_transcript_round(sf_transcript_t *t, ext_t s0, ext_t s1, ext_t s2) {
    uint8_t buf[32 + 48];
    ext_t r;
    memcpy(buf, t->state, 32);
    put_ext(buf + 32, s0);
    put_ext(buf + 48, s1);
    put_ext(buf + 64, s2);
    sha256(buf, sizeof buf, t->state);
    r.c0 = gl_canon(get_u64le(t->state));
    r.c1 = gl_canon(get_u64le(t->state + 8));
    return r;
}

/* ------------------------------------------------------------ accumulators */
typedef struct {
    acc128_t a[6]; /* s0.c0 s0.c1 s1.c0 s1.c1 s2.c0 s2.c1 */
} acc6_t;

typedef struct {
    ext_t s[3];
    uint64_t pad[2]; /* 64 bytes: one cache line per thread */
} sf_partial_t;

static SF_ALIGN64 sf_partial_t g_partial[SF_MAX_THREADS];

GL_INLINE void eval_pair(acc6_t *A, ext_t f0, ext_t f1, ext_t g0, ext_t g1) {
    ext_t p0 = ext_mul(f0, g0);
    ext_t p1 = ext_mul(f1, g1);
    ext_t p2 = ext_mul(ext_sub(ext_dbl(f1), f0), ext_sub(ext_dbl(g1), g0));
    acc_add(&A->a[0], p0.c0);
    acc_add(&A->a[1], p0.c1);
    acc_add(&A->a[2], p1.c0);
    acc_add(&A->a[3], p1.c1);
    acc_add(&A->a[4], p2.c0);
    acc_add(&A->a[5], p2.c1);
}

static void acc6_finish(const acc6_t *A, ext_t out[3]) {
    int k;
    for (k = 0; k < 3; k++) {
        out[k].c0 = acc_reduce(A->a[2 * k]);
        out[k].c1 = acc_reduce(A->a[2 * k + 1]);
    }
}

/* Evaluate pass over one contiguous block of M (even) elements. */
static void eval_block(const ext_t *f, const ext_t *g, int64_t M, acc6_t *A) {
    int64_t j;
    for (j = 0; j < M; j += 2) eval_pair(A, f[j], f[j + 1], g[j], g[j + 1]);
}

/*
 * Fused pass on one lane block: reads the level-(i-1) table (Mprev elements), folds it with r into the level-i table (Mprev/2 elements, written in place at the start of the block) and accumulates the round-i message on the fly.
*/
static void fold_eval_block(ext_t *f, ext_t *g, int64_t Mprev, ext_t r, acc6_t *A) {
    int64_t k, K = Mprev >> 2;
    for (k = 0; k < K; k++) {
        const ext_t f0 = f[4 * k], f1 = f[4 * k + 1], f2 = f[4 * k + 2], f3 = f[4 * k + 3];
        const ext_t g0 = g[4 * k], g1 = g[4 * k + 1], g2 = g[4 * k + 2], g3 = g[4 * k + 3];
        const ext_t a0 = ext_fold(f0, f1, r), a1 = ext_fold(f2, f3, r);
        const ext_t b0 = ext_fold(g0, g1, r), b1 = ext_fold(g2, g3, r);
        f[2 * k] = a0;
        f[2 * k + 1] = a1;
        g[2 * k] = b0;
        g[2 * k + 1] = b1;
        eval_pair(A, a0, a1, b0, b1);
    }
}

static ext_t sum_partials(int nth, int k) {
    ext_t s = {0, 0};
    int t;
    for (t = 0; t < nth; t++) s = ext_add(s, g_partial[t].s[k]);
    return s;
}

static void record_pass(sf_result_t *res, char kind, int64_t ticks) {
    if (res->npasses < SF_MAX_PASSES) {
        res->pass_kind[res->npasses] = kind;
        res->pass_ticks[res->npasses] = ticks;
    }
    res->npasses++;
}

static void record_round(sf_result_t *res, sf_transcript_t *tr, int n_global, int gi,
                         ext_t s0, ext_t s1, ext_t s2, ext_t *r_out) {
    if (gi == 0) {
        res->claim = ext_add(s0, s1); /* == inner(f, g) exactly */
        sf_transcript_init(tr, n_global, res->claim);
    }
    *r_out = sf_transcript_round(tr, s0, s1, s2);
    res->msgs[gi][0] = s0;
    res->msgs[gi][1] = s1;
    res->msgs[gi][2] = s2;
    res->chal[gi] = *r_out;
}

/* ----------------------------------------------------------- fused engine */
typedef struct {
    ext_t *f, *g;
    int64_t L, B;      /* lanes, elements per lane (power of two, >= 2) */
    int R;             /* rounds executed here = log2(B) */
    int i0;            /* global index of the first round */
    int n_global;
    int T, dyn;
    char kE, kF, kL;   /* pass-kind letters */
    sf_transcript_t *tr;
    sf_result_t *res;
    int threads_used;
} fused_job_t;

static void fused_lane_work(fused_job_t *J, int s, int64_t l, ext_t r, acc6_t *A) {
    ext_t *f = J->f + l * J->B, *g = J->g + l * J->B;
    if (s == 0) {
        eval_block(f, g, J->B, A);
    } else if (s < J->R) {
        fold_eval_block(f, g, J->B >> (s - 1), r, A);
    } else { /* final fold of the 2-element level R-1 table */
        f[0] = ext_fold(f[0], f[1], r);
        g[0] = ext_fold(g[0], g[1], r);
    }
}

static void run_fused(fused_job_t *J) {
    ext_t r_cur = {0, 0};
    int64_t t_mark = 0;
    int nth_used = 1;
    sf_barrier_reset(&g_barrier);
#ifdef _OPENMP
#pragma omp parallel num_threads(J->T) shared(r_cur, t_mark, nth_used)
#endif
    {
        const int tid = omp_get_thread_num();
        const int nth = omp_get_num_threads();
        const int64_t la = J->L * tid / nth, lb = J->L * (tid + 1) / nth;
        long sense = 0;
        int s;
        if (tid == 0) {
            nth_used = nth;
            t_mark = sf_ticks();
        }
        sf_barrier_wait(&g_barrier, nth, &sense);
        for (s = 0; s <= J->R; s++) {
            acc6_t A;
            const ext_t r = r_cur;
            memset(&A, 0, sizeof A);
            if (J->dyn) {
                int l;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1) nowait
#endif
                for (l = 0; l < (int)J->L; l++) fused_lane_work(J, s, l, r, &A);
            } else {
                int64_t l;
                for (l = la; l < lb; l++) fused_lane_work(J, s, l, r, &A);
            }
            if (s < J->R) acc6_finish(&A, g_partial[tid].s);
            sf_barrier_wait(&g_barrier, nth, &sense);
            if (tid == 0) {
                int64_t t1 = sf_ticks();
                record_pass(J->res, s == 0 ? J->kE : (s < J->R ? J->kF : J->kL), t1 - t_mark);
                if (s < J->R) {
                    ext_t s0 = sum_partials(nth, 0), s1 = sum_partials(nth, 1), s2 = sum_partials(nth, 2);
                    int64_t t2;
                    record_round(J->res, J->tr, J->n_global, J->i0 + s, s0, s1, s2, &r_cur);
                    t2 = sf_ticks();
                    J->res->transcript_ticks += t2 - t1;
                    t_mark = t2;
                }
            }
            sf_barrier_wait(&g_barrier, nth, &sense);
        }
    }
    J->threads_used = nth_used;
}

/* --------------------------------------------------------- unfused engine */
static int run_unfused(ext_t *f, ext_t *g, ext_t *bf, ext_t *bg, int n, int T,
                       sf_transcript_t *tr, sf_result_t *res) {
    ext_t r_cur = {0, 0};
    int64_t t_mark = 0;
    int nth_used = 1;
    sf_barrier_reset(&g_barrier);
#ifdef _OPENMP
#pragma omp parallel num_threads(T) shared(r_cur, t_mark, nth_used)
#endif
    {
        const int tid = omp_get_thread_num();
        const int nth = omp_get_num_threads();
        ext_t *sf_ = f, *sg = g, *df = bf, *dg = bg, *tmp;
        int64_t M = (int64_t)1 << n;
        long sense = 0;
        int i;
        if (tid == 0) {
            nth_used = nth;
            t_mark = sf_ticks();
        }
        sf_barrier_wait(&g_barrier, nth, &sense);
        for (i = 0; i < n; i++) {
            const int64_t P = M >> 1;
            const int64_t pa = P * tid / nth, pb = P * (tid + 1) / nth;
            int64_t j;
            acc6_t A;
            ext_t r;
            memset(&A, 0, sizeof A);
            /* evaluate pass */
            eval_block(sf_ + 2 * pa, sg + 2 * pa, 2 * (pb - pa), &A);
            acc6_finish(&A, g_partial[tid].s);
            sf_barrier_wait(&g_barrier, nth, &sense);
            if (tid == 0) {
                int64_t t1 = sf_ticks(), t2;
                ext_t s0 = sum_partials(nth, 0), s1 = sum_partials(nth, 1), s2 = sum_partials(nth, 2);
                record_pass(res, 'E', t1 - t_mark);
                record_round(res, tr, n, i, s0, s1, s2, &r_cur);
                t2 = sf_ticks();
                res->transcript_ticks += t2 - t1;
                t_mark = t2;
            }
            sf_barrier_wait(&g_barrier, nth, &sense);
            /* out-of-place fold pass */
            r = r_cur;
            for (j = pa; j < pb; j++) {
                df[j] = ext_fold(sf_[2 * j], sf_[2 * j + 1], r);
                dg[j] = ext_fold(sg[2 * j], sg[2 * j + 1], r);
            }
            sf_barrier_wait(&g_barrier, nth, &sense);
            if (tid == 0) {
                int64_t t1 = sf_ticks();
                record_pass(res, 'U', t1 - t_mark);
                t_mark = t1;
            }
            tmp = sf_; sf_ = df; df = tmp;
            tmp = sg; sg = dg; dg = tmp;
            M = P;
        }
        if (tid == 0) {
            res->final_f = sf_[0];
            res->final_g = sg[0];
        }
    }
    return nth_used;
}

/* ------------------------------------------------------------------ driver */
int sf_auto_lanes(int n, int threads) {
    int64_t L = 1, cap;
    if (threads <= 1 || n < 2) return 1;
    while (L < threads) L <<= 1;
    L *= 16; /* >= 16 lanes per thread: static imbalance <= 1/16 of a thread's work */
    cap = (int64_t)1 << (n - 1);
    if (L > cap) L = cap;
    return (int)L;
}

int sf_prove(const sf_config_t *cfg, ext_t *f, ext_t *g, ext_t *bf, ext_t *bg, sf_result_t *res) {
    const int n = cfg->n;
    int T = cfg->threads;
    sf_transcript_t tr;
    int64_t t0, N;
    memset(res, 0, sizeof *res);
    if (n < 1 || n >= SF_MAX_LOGN) return -1;
    if (T < 1) T = 1;
    if (T > SF_MAX_THREADS) T = SF_MAX_THREADS;
    N = (int64_t)1 << n;
    t0 = sf_ticks();
    if (cfg->variant == SF_FUSED) {
        int64_t L = cfg->lanes > 0 ? cfg->lanes : sf_auto_lanes(n, T);
        int lg = 0;
        fused_job_t J;
        if ((L & (L - 1)) != 0 || L > N / 2) return -2;
        while (((int64_t)1 << lg) < L) lg++;
        memset(&J, 0, sizeof J);
        J.f = f; J.g = g; J.L = L; J.B = N / L; J.R = n - lg; J.i0 = 0; J.n_global = n;
        J.T = T; J.dyn = cfg->sched_dynamic; J.kE = 'E'; J.kF = 'F'; J.kL = 'L';
        J.tr = &tr; J.res = res;
        run_fused(&J);
        res->lanes_used = (int)L;
        res->threads_used = J.threads_used;
        if (L > 1) { /* host tail: last log2(L) rounds on the L lane residues */
            int64_t tt0 = sf_ticks(), l, B = N / L;
            fused_job_t H;
            for (l = 1; l < L; l++) { /* ascending in-place gather: l < l*B, slot l already consumed */
                f[l] = f[l * B];
                g[l] = g[l * B];
            }
            record_pass(res, 'g', sf_ticks() - tt0);
            memset(&H, 0, sizeof H);
            H.f = f; H.g = g; H.L = 1; H.B = L; H.R = lg; H.i0 = n - lg; H.n_global = n;
            H.T = 1; H.dyn = 0; H.kE = 'e'; H.kF = 'f'; H.kL = 'l';
            H.tr = &tr; H.res = res;
            run_fused(&H);
            res->tail_ticks = sf_ticks() - tt0;
        }
        res->final_f = f[0];
        res->final_g = g[0];
    } else {
        if (bf == NULL || bg == NULL) return -3;
        res->lanes_used = 0;
        res->threads_used = run_unfused(f, g, bf, bg, n, T, &tr, res);
    }
    res->total_ticks = sf_ticks() - t0;
    return 0;
}
