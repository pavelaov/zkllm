/*
 * gl_field.h -- Goldilocks base field and its quadratic extension (C port). Bit-exact port of src/edgecert/field.py (golden model, spec/sumfold_kernel.md sec. 2): p   = 2^64 - 2^32 + 1 Ext = F_p[X] / (X^2 - 7), element (c0, c1) = c0 + c1*X, stored as {c1, c0}. All values are canonical (< p) at every function boundary. Compilers: gcc/clang use unsigned __int128; MSVC uses _umul128 / _addcarry_u64.
*/
#ifndef EDGECERT_GL_FIELD_H
#define EDGECERT_GL_FIELD_H

#include <stdint.h>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#pragma intrinsic(_umul128)
#define GL_INLINE static __forceinline
#define GL_MSVC 1
#else
#define GL_INLINE static inline __attribute__((always_inline))
#define GL_MSVC 0
#endif

#define GL_P   0xFFFFFFFF00000001ULL
#define GL_EPS 0x00000000FFFFFFFFULL /* 2^64 mod p = 2^32 - 1 */
#define GL_W   7ULL                  /* extension non-residue */

typedef struct {
    uint64_t c0; /* low word  */
    uint64_t c1; /* high word */
} ext_t;

/* ---------------------------------------------------------------- base field */

/* Branch-free masks: the conditional corrections below depend on random data, so
 * they are written with all-ones/zero masks instead of branches (same values). */
#define GL_MASK(cond) ((uint64_t)0 - (uint64_t)(cond))

GL_INLINE uint64_t gl_add(uint64_t a, uint64_t b) {
    /* field.fadd: s = a + b; s - p if s >= p.  a + b < 2p < 2^65.
     * carry out: true value s + 2^64, minus p == s + EPS (then already < p). */
    uint64_t s = a + b;
    s += GL_EPS & GL_MASK(s < a);
    s -= GL_P & GL_MASK(s >= GL_P);
    return s;
}

GL_INLINE uint64_t gl_sub(uint64_t a, uint64_t b) {
    /* field.fsub: a - b if a >= b else a - b + p.  (+p == -EPS mod 2^64) */
    uint64_t d = a - b;
    d -= GL_EPS & GL_MASK(a < b);
    return d;
}

/*
 * field.reduce128, written exactly as the RTL does it (no multiplier): x = hi*2^64 + lo, hi = hh*2^32 + hl x = lo - hh + hl*(2^32 - 1)  (mod p)
*/
GL_INLINE uint64_t gl_reduce128(uint64_t lo, uint64_t hi) {
    uint64_t hh = hi >> 32;
    uint64_t hl = hi & 0xFFFFFFFFULL;
    uint64_t t = lo - hh;
    t -= GL_EPS & GL_MASK(lo < hh);           /* borrow: add p (mod 2^64) */
    uint64_t u = (hl << 32) - hl;             /* hl * (2^32 - 1), shift-subtract */
    uint64_t s = t + u;
    s += GL_EPS & GL_MASK(s < t);             /* value >= 2^64: subtract p once; value < 2p */
    s -= GL_P & GL_MASK(s >= GL_P);           /* so at most one correction applies */
    return s;
}

GL_INLINE uint64_t gl_mul(uint64_t a, uint64_t b) {
#if GL_MSVC
    uint64_t hi;
    uint64_t lo = _umul128(a, b, &hi);
    return gl_reduce128(lo, hi);
#else
    unsigned __int128 x = (unsigned __int128)a * b;
    return gl_reduce128((uint64_t)x, (uint64_t)(x >> 64));
#endif
}

/* 7*t as (t << 3) - t mod p (spec sec. 2); computed via a 67-bit value. */
GL_INLINE uint64_t gl_mul7(uint64_t t) {
    uint64_t lo8 = t << 3;
    uint64_t hi8 = t >> 61;
    uint64_t t8 = gl_reduce128(lo8, hi8);     /* 8t mod p (hh = 0) */
    return gl_sub(t8, t);
}

GL_INLINE uint64_t gl_canon(uint64_t x) { return x >= GL_P ? x - GL_P : x; }

/* ------------------------------------------------------------ extension */

GL_INLINE ext_t ext_add(ext_t a, ext_t b) {
    ext_t r; r.c0 = gl_add(a.c0, b.c0); r.c1 = gl_add(a.c1, b.c1); return r;
}

GL_INLINE ext_t ext_sub(ext_t a, ext_t b) {
    ext_t r; r.c0 = gl_sub(a.c0, b.c0); r.c1 = gl_sub(a.c1, b.c1); return r;
}

GL_INLINE ext_t ext_dbl(ext_t a) { return ext_add(a, a); }

/* Karatsuba (field.emul): 3 base multiplications + multiply-by-7. */
GL_INLINE ext_t ext_mul(ext_t a, ext_t b) {
    uint64_t t0 = gl_mul(a.c0, b.c0);
    uint64_t t1 = gl_mul(a.c1, b.c1);
    uint64_t t2 = gl_mul(gl_add(a.c0, a.c1), gl_add(b.c0, b.c1));
    ext_t r;
    r.c0 = gl_add(t0, gl_mul7(t1));
    r.c1 = gl_sub(gl_sub(t2, t0), t1);
    return r;
}

/* fold: v0 + r * (v1 - v0)  (sumfold.fold) */
GL_INLINE ext_t ext_fold(ext_t v0, ext_t v1, ext_t r) {
    return ext_add(v0, ext_mul(r, ext_sub(v1, v0)));
}

/* * ------------------------------------------------ lazy 128-bit accumulator Sum of canonical base elements kept as an exact 128-bit integer and reduced once at the end with gl_reduce128.  Exact (sum < 2^64 * count < 2^128), so the result equals the field sum computed with gl_add in any order.
*/
typedef struct { uint64_t lo, hi; } acc128_t;

GL_INLINE void acc_add(acc128_t *a, uint64_t x) {
#if GL_MSVC
    unsigned char c = _addcarry_u64(0, a->lo, x, &a->lo);
    a->hi += c;
#else
    uint64_t s = a->lo + x;
    a->hi += (s < x);
    a->lo = s;
#endif
}

GL_INLINE uint64_t acc_reduce(acc128_t a) { return gl_reduce128(a.lo, a.hi); }

#endif /* EDGECERT_GL_FIELD_H */
