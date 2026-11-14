/*
 * sumfold.h -- CPU SumFold prover (product sumcheck over Goldilocks / Ext2). Semantics: spec/sumfold_kernel.md (v1) and src/edgecert/sumfold.py::prove(). Two variants share the same field code and transcript: SF_FUSED   : spec sec. 3.1 fused fold-and-evaluate pass, in place, lane-local contiguous blocks (spec sec. 3.2), host tail for the last log2(L) rounds. SF_UNFUSED : naive baseline, per round one evaluate pass then one out-of-place fold pass into a ping-pong buffer. Both return the identical transcript (claim, round messages, challenges, finals).
*/
#ifndef EDGECERT_SUMFOLD_H
#define EDGECERT_SUMFOLD_H

#include <stdint.h>
#include "gl_field.h"

#define SF_MAX_LOGN   40
#define SF_MAX_PASSES (4 * SF_MAX_LOGN + 8)
#define SF_MAX_THREADS 256

typedef enum { SF_FUSED = 0, SF_UNFUSED = 1 } sf_variant_t;

typedef struct {
    uint8_t state[32];
} sf_transcript_t;

void  sf_transcript_init(sf_transcript_t *t, int n, ext_t claim);
ext_t sf_transcript_round(sf_transcript_t *t, ext_t s0, ext_t s1, ext_t s2);

typedef struct {
    int variant;        /* sf_variant_t */
    int n;              /* log2 N */
    int threads;        /* requested OpenMP threads (>= 1) */
    int lanes;          /* fused only: power of two, 1 <= lanes <= N/2; 0 = auto */
    int sched_dynamic;  /* fused only: 0 = static contiguous lane groups, 1 = dynamic lanes */
} sf_config_t;

typedef struct {
    ext_t claim;
    ext_t msgs[SF_MAX_LOGN][3];
    ext_t chal[SF_MAX_LOGN];
    ext_t final_f, final_g;
    int lanes_used;
    int threads_used;
    int npasses;
    /* pass kinds: 'E' evaluate, 'F' fused fold+evaluate, 'L' fold only (last fold),
     * 'U' out-of-place fold (unfused), 'G' gather lane residues; lower case = host tail */
    char pass_kind[SF_MAX_PASSES];
    int64_t pass_ticks[SF_MAX_PASSES];
    int64_t transcript_ticks;
    int64_t tail_ticks;
    int64_t total_ticks;
} sf_result_t;

/* Timer (QueryPerformanceCounter on Windows, CLOCK_MONOTONIC elsewhere). */
int64_t sf_ticks(void);
double  sf_tick_seconds(void);

int sf_auto_lanes(int n, int threads);

/*
 * Run the whole protocol exactly like sumfold.prove(f, g). f, g        : Ext tables of length N = 2^n (destroyed). bf, bg      : unfused only, scratch of length max(N/2, 1) each (may be NULL for fused). Returns 0 on success.
*/
int sf_prove(const sf_config_t *cfg, ext_t *f, ext_t *g, ext_t *bf, ext_t *bg, sf_result_t *res);

#endif
