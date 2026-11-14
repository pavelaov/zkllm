/*
 * mle.h -- multilinear-extension (MLE) partial evaluation by repeated folds, a bound product sumcheck and the eq-table verifier helpers used by the TRANSPARENT_REFERENCE matmul prover (src/edgecert/prover_adapters/matmul_sumcheck.py, bench/runners/gpt2_linear_prove.py). Built as a DLL together with the unmodified SumFold sources (sumfold.c, sha256.c), see Makefile.mle / build_mle_dll.bat.  gcc/clang only (unsigned __int128). Index conventions (identical to src/edgecert/sumfold.py and src/edgecert/verifier/mle.py): variables are bound LSB-first, fold(v, r)[j] = v[2j] + r * (v[2j+1] - v[2j]); a table of m < 2^nr real entries is implicitly zero-padded to 2^nr.  A fold of an integer table lifts every entry v to v mod p first (field.lift_signed), so the first fold is base x ext. Nothing here is zero knowledge or succinct (label TRANSPARENT_REFERENCE).
*/
#ifndef EDGECERT_MLE_H
#define EDGECERT_MLE_H

#include <stdint.h>
#include "gl_field.h"
#include "sumfold.h"

/* element types of an input matrix */
#define MLE_I8  0
#define MLE_U8  1
#define MLE_I16 2
#define MLE_U16 3
#define MLE_I32 4
#define MLE_I64 5
#define MLE_EXT 6   /* ext_t {c0, c1}, canonical */

#define MLE_MAX_VARS 40

typedef struct {
    int32_t n;                          /* number of sumcheck rounds (log2 of the padded length) */
    int32_t pad_;
    ext_t claim;                        /* H = sum_k f[k] g[k] */
    ext_t msgs[MLE_MAX_VARS][3];        /* (s0, s1, s2) per round */
    ext_t chal[MLE_MAX_VARS];           /* round challenges r_i */
    ext_t final_f, final_g;             /* f~(r), g~(r) */
} mle_sc_proof_t;

typedef struct {
    double fold_a_s;                    /* A~(r1, .) partial evaluation */
    double fold_w_s;                    /* W~(., r2) partial evaluation */
    double sumcheck_s;                  /* zero padding + product sumcheck (incl. SHA-256 transcript) */
    double total_s;
    int64_t table_bytes;                /* sumcheck tables f, g: 2 * 2^kappa * 16 bytes */
    int32_t kappa;                      /* sumcheck rounds */
    int32_t threads_used;
} mm_times_t;

/* build / version information (compile flags string) */
const char *mle_build_info(void);

/* Lift n integers (stride in elements) to canonical base-field elements. */
int mle_lift(const void *M, int dtype, int64_t n, int64_t stride, uint64_t *dst);

/*
 * Partial evaluation along one axis:  out[k] = sum_i M[i][k] eq(i, r)  for k < nkeep, computed by nr folds (LSB-first) of the zero-padded length-2^nr column, exactly as repeated sumfold.fold().  Element (i, k) is at M[i*sf + k*sk] (strides in elements). Requires nfold <= 2^nr.  threads > 1 uses OpenMP (small problems run serially).
*/
int mle_fold_axis(const void *M, int dtype, int64_t nfold, int64_t nkeep, int64_t sf, int64_t sk,
                  const ext_t *r, int nr, ext_t *out, int threads);

/*
 * Product sumcheck over f, g (length 2^n, destroyed). engine 0: single-thread fused fold-and-evaluate engine (spec 3.1, in place) with an optional bound transcript (bind = 1): state_0' = SHA256("EDGE-CERT/sumfold-bind/v1" || state_0 || u32le(ctx_len) || ctx || u32le(label_len) || label) i.e. edgecert.crypto.transcript.BoundSumcheckTranscript; bind = 0 is the golden sumfold.prove() transcript. engine 1: the existing SumFold engine sf_prove() (sumfold.c, fused, OpenMP), golden transcript only (bind must be 0).
*/
int mle_sumcheck(ext_t *f, ext_t *g, int n, const uint8_t *ctx, int ctx_len, const uint8_t *label,
                 int label_len, int bind, int engine, int threads, mle_sc_proof_t *out);

/*
 * Transparent matmul prover core for C = A W (A: T x K, W: K x N): a[k] = A~(r1, k)   (fold A along T with r1, t = |r1| rounds, T <= 2^t) w[k] = W~(k, r2)   (fold W along N with r2, nb = |r2| rounds, N <= 2^nb) bound product sumcheck over k (K zero-padded to 2^kappa, kappa = ceil(log2 K) >= 1). A[i][k] at A + i*sAt + k*sAk, W[k][j] at W + k*sWk + j*sWn (strides in elements). a_out / w_out (length K) receive a and w when non-NULL.
*/
int mm_prove(const void *A, int dtA, int64_t T, int64_t K, int64_t sAt, int64_t sAk,
             const void *W, int dtW, int64_t N, int64_t sWk, int64_t sWn,
             const ext_t *r1, int t, const ext_t *r2, int nb,
             const uint8_t *ctx, int ctx_len, const uint8_t *label, int label_len,
             int threads, mle_sc_proof_t *out, mm_times_t *tm, ext_t *a_out, ext_t *w_out);

/* * ---------------------------------------------------------------- verifier side Independent algorithm (eq-table contraction with lazy exact 128-bit accumulation), no fold code is shared with the prover.
*/

/* eq table of 2^nr entries, same order as verifier.mle.eq_table (LSB-first). */
int mlev_eq_table(const ext_t *r, int nr, ext_t *out);

/* out = sum_{i<R, j<C} M[i][j] eqR[i] eqC[j],  M[i][j] at M[i*sR + j*sC].  Integer dtypes
 * only (MLE_I64 entries must satisfy |v| < 2^63).                                        */
int mlev_bilinear(const void *M, int dtype, int64_t R, int64_t C, int64_t sR, int64_t sC,
                  const ext_t *eqR, const ext_t *eqC, int threads, ext_t *out);

/* scratch accounting (bytes malloc'd by this library) */
int64_t mle_scratch_peak(void);
int64_t mle_scratch_current(void);
void mle_scratch_reset(void);

#endif
