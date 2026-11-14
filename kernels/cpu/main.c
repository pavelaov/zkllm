/*
 * main.c -- sumfold_cpu CLI: generate the spec 3.3 test vectors, run the CPU SumFold prover (fused or unfused) W + R times, and write one JSON record with all round messages, challenges, finals (128-bit hex {c1,c0}) and per-pass / total timings. sumfold_cpu --logn n --threads t --variant fused|unfused --seedf HEX --seedg HEX --reps R --warmup W --out file.json [--lanes L] [--sched static|dynamic] [--affinity HEXMASK] sumfold_cpu --selftest --affinity restricts the process to the given logical-CPU mask (Windows processor group 0) before any thread is created; the runner uses it to pick P-cores first on the hybrid i9-14900K.  Without it, thread placement is left to the OS.
*/
#if defined(__MINGW32__) && !defined(__USE_MINGW_ANSI_STDIO)
#define __USE_MINGW_ANSI_STDIO 1 /* C99 printf (%llx) on MinGW */
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sumfold.h"
#include "sha256.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#ifndef SF_BUILD_FLAGS
#define SF_BUILD_FLAGS "unknown"
#endif

#define GL_MASK64 0xFFFFFFFFFFFFFFFFULL

static uint64_t xorshift64(uint64_t x) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}

/* sumfold.gen_vector: v[i] = x_{i+1} mod p, x_0 = seed, c1 = 0. */
static void gen_vector(uint64_t seed, uint64_t *out, int64_t N) {
    uint64_t x = seed;
    int64_t i;
    for (i = 0; i < N; i++) {
        x = xorshift64(x);
        out[i] = x >= GL_P ? x - GL_P : x;
    }
}

static void *xaligned_alloc(size_t bytes) {
#ifdef _WIN32
    return _aligned_malloc(bytes, 64);
#else
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
    return p;
#endif
}

static void xaligned_free(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

static void ehex(char *buf, ext_t e) { /* {c1, c0}, like field.ehex */
    sprintf(buf, "%016llx%016llx", (unsigned long long)e.c1, (unsigned long long)e.c0);
}

static int parse_hex64(const char *s, uint64_t *out) {
    char *end = NULL;
    unsigned long long v;
    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) s += 2;
    v = strtoull(s, &end, 16);
    if (end == s || *end != '\0') return -1;
    *out = (uint64_t)v;
    return 0;
}

static int same_result(const sf_result_t *a, const sf_result_t *b, int n) {
    int i, k;
    if (memcmp(&a->claim, &b->claim, sizeof(ext_t)) || memcmp(&a->final_f, &b->final_f, sizeof(ext_t)) ||
        memcmp(&a->final_g, &b->final_g, sizeof(ext_t)))
        return 0;
    for (i = 0; i < n; i++) {
        if (memcmp(&a->chal[i], &b->chal[i], sizeof(ext_t))) return 0;
        for (k = 0; k < 3; k++)
            if (memcmp(&a->msgs[i][k], &b->msgs[i][k], sizeof(ext_t))) return 0;
    }
    return 1;
}

/* ---------------------------------------------------------------- selftest */
static int selftest(void) {
#if !GL_MSVC
    uint64_t s = 0x9E3779B97F4A7C15ULL;
    int i, bad = 0;
    for (i = 0; i < 2000000; i++) {
        uint64_t a, b, c, d;
        unsigned __int128 x;
        s = xorshift64(s); a = s;
        s = xorshift64(s); b = s;
        if (i & 1) { a %= GL_P; b %= GL_P; }
        else { a = GL_P - 1 - (a & 0xFFFF); b = GL_P - 1 - (b & 0xFFFF); } /* edge region */
        x = (unsigned __int128)a * b;
        if (gl_mul(a, b) != (uint64_t)(x % GL_P)) bad++;
        s = xorshift64(s); c = s;
        s = xorshift64(s); d = s;
        x = ((unsigned __int128)c << 64) | d;
        if (gl_reduce128(d, c) != (uint64_t)(x % GL_P)) bad++;
        if (gl_add(a, b) != (uint64_t)(((unsigned __int128)a + b) % GL_P)) bad++;
        if (gl_sub(a, b) != (uint64_t)(((unsigned __int128)a + GL_P - b) % GL_P)) bad++;
        if (gl_mul7(a) != (uint64_t)(((unsigned __int128)a * 7) % GL_P)) bad++;
    }
    /* reduce128 extremes */
    {
        uint64_t his[] = {0, 1, 0xFFFFFFFFULL, 0xFFFFFFFF00000000ULL, GL_MASK64, 0xFFFFFFFEFFFFFFFFULL};
        uint64_t los[] = {0, 1, GL_P - 1, GL_P, GL_MASK64, 0xFFFFFFFFULL};
        int a, b;
        for (a = 0; a < 6; a++)
            for (b = 0; b < 6; b++) {
                unsigned __int128 x = ((unsigned __int128)his[a] << 64) | los[b];
                if (gl_reduce128(los[b], his[a]) != (uint64_t)(x % GL_P)) bad++;
            }
    }
    {   /* SHA-256("abc") */
        static const uint8_t want[32] = {0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40,
                                         0xde, 0x5d, 0xae, 0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17,
                                         0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad};
        uint8_t h[32];
        sha256("abc", 3, h);
        if (memcmp(h, want, 32)) bad++;
    }
    printf("{\"selftest\": \"%s\", \"errors\": %d}\n", bad ? "FAIL" : "OK", bad);
    return bad ? 1 : 0;
#else
    printf("{\"selftest\": \"SKIPPED_MSVC\"}\n");
    return 0;
#endif
}

static void usage(void) {
    fprintf(stderr,
            "usage: sumfold_cpu --logn n --threads t --variant fused|unfused --seedf HEX --seedg HEX\n"
            "                   --reps R --warmup W --out file.json [--lanes L] [--sched static|dynamic]\n"
            "                   [--affinity HEXMASK]\n"
            "       sumfold_cpu --selftest\n");
}

int main(int argc, char **argv) {
    int n = 10, threads = 1, variant = SF_FUSED, reps = 1, warmup = 0, lanes = 0, dyn = 0;
    uint64_t seedf = 1, seedg = 2, affinity = 0;
    const char *out_path = NULL;
    int i, rc;
    int64_t N;
    uint64_t *f0, *g0;
    ext_t *f, *g, *bf = NULL, *bg = NULL;
    sf_result_t *res, *first;
    sf_config_t cfg;
    int consistent = 1;
    double tick_s = sf_tick_seconds();
    int64_t reset_ticks_total = 0;
    FILE *fp;
    char hb[40];

    for (i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (!strcmp(a, "--selftest")) return selftest();
        if (v == NULL) { usage(); return 2; }
        if (!strcmp(a, "--logn")) n = atoi(v);
        else if (!strcmp(a, "--threads")) threads = atoi(v);
        else if (!strcmp(a, "--variant")) {
            if (!strcmp(v, "fused")) variant = SF_FUSED;
            else if (!strcmp(v, "unfused")) variant = SF_UNFUSED;
            else { usage(); return 2; }
        } else if (!strcmp(a, "--seedf")) { if (parse_hex64(v, &seedf)) { usage(); return 2; } }
        else if (!strcmp(a, "--seedg")) { if (parse_hex64(v, &seedg)) { usage(); return 2; } }
        else if (!strcmp(a, "--reps")) reps = atoi(v);
        else if (!strcmp(a, "--warmup")) warmup = atoi(v);
        else if (!strcmp(a, "--out")) out_path = v;
        else if (!strcmp(a, "--lanes")) lanes = atoi(v);
        else if (!strcmp(a, "--affinity")) { if (parse_hex64(v, &affinity)) { usage(); return 2; } }
        else if (!strcmp(a, "--sched")) {
            if (!strcmp(v, "static")) dyn = 0;
            else if (!strcmp(v, "dynamic")) dyn = 1;
            else { usage(); return 2; }
        } else { usage(); return 2; }
        i++;
    }
    if (n < 1 || n > 30 || threads < 1 || reps < 1 || warmup < 0 || seedf == 0 || seedg == 0) {
        fprintf(stderr, "invalid arguments (need 1<=logn<=30, threads>=1, reps>=1, non-zero seeds)\n");
        return 2;
    }
    if (affinity != 0) {
#ifdef _WIN32
        if (!SetProcessAffinityMask(GetCurrentProcess(), (DWORD_PTR)affinity)) {
            fprintf(stderr, "SetProcessAffinityMask(0x%llx) failed\n", (unsigned long long)affinity);
            return 2;
        }
#else
        fprintf(stderr, "--affinity is only implemented on Windows\n");
        return 2;
#endif
    }
    N = (int64_t)1 << n;
    f0 = (uint64_t *)xaligned_alloc((size_t)N * sizeof(uint64_t));
    g0 = (uint64_t *)xaligned_alloc((size_t)N * sizeof(uint64_t));
    f = (ext_t *)xaligned_alloc((size_t)N * sizeof(ext_t));
    g = (ext_t *)xaligned_alloc((size_t)N * sizeof(ext_t));
    if (variant == SF_UNFUSED) {
        int64_t H = N / 2 > 0 ? N / 2 : 1;
        bf = (ext_t *)xaligned_alloc((size_t)H * sizeof(ext_t));
        bg = (ext_t *)xaligned_alloc((size_t)H * sizeof(ext_t));
        if (!bf || !bg) { fprintf(stderr, "out of memory\n"); return 3; }
    }
    res = (sf_result_t *)calloc(1, sizeof(sf_result_t));
    first = (sf_result_t *)calloc(1, sizeof(sf_result_t));
    if (!f0 || !g0 || !f || !g || !res || !first) { fprintf(stderr, "out of memory\n"); return 3; }
    gen_vector(seedf, f0, N);
    gen_vector(seedg, g0, N);

    cfg.variant = variant; cfg.n = n; cfg.threads = threads; cfg.lanes = lanes; cfg.sched_dynamic = dyn;

    fp = out_path ? fopen(out_path, "w") : stdout;
    if (!fp) { fprintf(stderr, "cannot open %s\n", out_path); return 4; }

    /* ---- runs: warmup + timed reps; tables are re-expanded (untimed) before each run */
    {
        int total = warmup + reps, run, p;
        double *tot = (double *)calloc((size_t)total, sizeof(double));
        double *trs = (double *)calloc((size_t)total, sizeof(double));
        double *tail = (double *)calloc((size_t)total, sizeof(double));
        double *pass = (double *)calloc((size_t)total * SF_MAX_PASSES, sizeof(double));
        int npasses = 0;
        for (run = 0; run < total; run++) {
            int64_t tr0 = sf_ticks(), j;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
            for (j = 0; j < N; j++) {
                f[j].c0 = f0[j]; f[j].c1 = 0;
                g[j].c0 = g0[j]; g[j].c1 = 0;
            }
            reset_ticks_total += sf_ticks() - tr0;
            rc = sf_prove(&cfg, f, g, bf, bg, res);
            if (rc != 0) { fprintf(stderr, "sf_prove failed rc=%d (lanes must be a power of two <= N/2)\n", rc); return 5; }
            if (run == 0) memcpy(first, res, sizeof *res);
            else if (!same_result(first, res, n)) consistent = 0;
            tot[run] = res->total_ticks * tick_s;
            trs[run] = res->transcript_ticks * tick_s;
            tail[run] = res->tail_ticks * tick_s;
            npasses = res->npasses < SF_MAX_PASSES ? res->npasses : SF_MAX_PASSES;
            for (p = 0; p < npasses; p++) pass[(size_t)run * SF_MAX_PASSES + p] = res->pass_ticks[p] * tick_s;
        }

        /* ---- JSON */
        fprintf(fp, "{\n");
        fprintf(fp, "  \"impl\": \"cpu_c\",\n  \"variant\": \"%s\",\n", variant == SF_FUSED ? "fused" : "unfused");
        fprintf(fp, "  \"n\": %d,\n  \"n_elems\": %lld,\n", n, (long long)N);
        fprintf(fp, "  \"threads_requested\": %d,\n  \"threads_used\": %d,\n", threads, first->threads_used);
        fprintf(fp, "  \"lanes\": %d,\n  \"sched\": \"%s\",\n", first->lanes_used, dyn ? "dynamic" : "static");
        if (affinity) fprintf(fp, "  \"affinity_mask\": \"%llx\",\n", (unsigned long long)affinity);
        else fprintf(fp, "  \"affinity_mask\": null,\n");
        fprintf(fp, "  \"seed_f\": \"%016llx\",\n  \"seed_g\": \"%016llx\",\n",
                (unsigned long long)seedf, (unsigned long long)seedg);
        ehex(hb, first->claim);
        fprintf(fp, "  \"claim\": \"%s\",\n", hb);
        fprintf(fp, "  \"msgs\": [");
        for (i = 0; i < n; i++) {
            char h0[40], h1[40], h2[40];
            ehex(h0, first->msgs[i][0]); ehex(h1, first->msgs[i][1]); ehex(h2, first->msgs[i][2]);
            fprintf(fp, "%s[\"%s\", \"%s\", \"%s\"]", i ? ", " : "", h0, h1, h2);
        }
        fprintf(fp, "],\n  \"challenges\": [");
        for (i = 0; i < n; i++) {
            ehex(hb, first->chal[i]);
            fprintf(fp, "%s\"%s\"", i ? ", " : "", hb);
        }
        ehex(hb, first->final_f);
        fprintf(fp, "],\n  \"final_f\": \"%s\",\n", hb);
        ehex(hb, first->final_g);
        fprintf(fp, "  \"final_g\": \"%s\",\n", hb);
        fprintf(fp, "  \"consistent_across_runs\": %s,\n", consistent ? "true" : "false");
        fprintf(fp, "  \"warmup\": %d,\n  \"reps\": %d,\n", warmup, reps);
        fprintf(fp, "  \"timer\": \"%s\",\n  \"tick_seconds\": %.6e,\n",
#ifdef _WIN32
                "QueryPerformanceCounter",
#else
                "clock_gettime(CLOCK_MONOTONIC)",
#endif
                tick_s);
        fprintf(fp, "  \"reset_s_total_untimed\": %.9f,\n", reset_ticks_total * tick_s);
        fprintf(fp, "  \"pass_kinds\": \"");
        for (p = 0; p < npasses; p++) fputc(first->pass_kind[p], fp);
        fprintf(fp, "\",\n");
        fprintf(fp, "  \"total_s\": [");
        for (run = warmup; run < total; run++) fprintf(fp, "%s%.9f", run > warmup ? ", " : "", tot[run]);
        fprintf(fp, "],\n  \"transcript_s\": [");
        for (run = warmup; run < total; run++) fprintf(fp, "%s%.9f", run > warmup ? ", " : "", trs[run]);
        fprintf(fp, "],\n  \"tail_s\": [");
        for (run = warmup; run < total; run++) fprintf(fp, "%s%.9f", run > warmup ? ", " : "", tail[run]);
        fprintf(fp, "],\n  \"warmup_total_s\": [");
        for (run = 0; run < warmup; run++) fprintf(fp, "%s%.9f", run ? ", " : "", tot[run]);
        fprintf(fp, "],\n  \"pass_s\": [");
        for (run = warmup; run < total; run++) {
            fprintf(fp, "%s[", run > warmup ? ",\n    " : "\n    ");
            for (p = 0; p < npasses; p++) fprintf(fp, "%s%.9f", p ? ", " : "", pass[(size_t)run * SF_MAX_PASSES + p]);
            fprintf(fp, "]");
        }
        fprintf(fp, "],\n");
#if defined(__GNUC__) && !defined(__clang__)
        fprintf(fp, "  \"compiler\": \"gcc %s\",\n", __VERSION__);
#elif defined(_MSC_VER)
        fprintf(fp, "  \"compiler\": \"msvc %d\",\n", _MSC_FULL_VER);
#else
        fprintf(fp, "  \"compiler\": \"unknown\",\n");
#endif
        fprintf(fp, "  \"build_flags\": \"%s\",\n", SF_BUILD_FLAGS);
#ifdef _OPENMP
        fprintf(fp, "  \"openmp\": %d,\n", (int)_OPENMP);
#else
        fprintf(fp, "  \"openmp\": null,\n");
#endif
        fprintf(fp, "  \"bytes_per_ext\": %d\n}\n", (int)sizeof(ext_t));
        free(tot); free(trs); free(tail); free(pass);
    }
    if (out_path) fclose(fp);
    xaligned_free(f0); xaligned_free(g0); xaligned_free(f); xaligned_free(g);
    if (bf) xaligned_free(bf);
    if (bg) xaligned_free(bg);
    free(res); free(first);
    return consistent ? 0 : 6;
}
