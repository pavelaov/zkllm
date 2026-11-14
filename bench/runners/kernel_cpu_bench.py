"""CPU SumFold benchmark: fused vs unfused, thread scaling, bit-exactness.

Default sweep (the task's full grid):
    logN in {12,14,16,18,20,22,24}, threads in {1,8,24,32}, variants {fused, unfused},
    3 warm-up + 10 timed repetitions per configuration (inside one process; the tables
    are re-expanded, untimed, before every repetition).

Examples (from the repository root):
    python bench/runners/kernel_cpu_bench.py                        # full grid, gcc
    python bench/runners/kernel_cpu_bench.py --reduced              # reduced grid
    python bench/runners/kernel_cpu_bench.py --compilers gcc,msvc --out results/raw/kernels/cpu_bench.json
    python bench/runners/kernel_cpu_bench.py --auto-reduce          # reduced grid if Vivado/xsim is running

Every configuration records median, p95 and a bootstrap 95% CI of the median of the end-to-end prover time (all passes + transcript + host tail; data generation excluded), per-pass medians, and whether the transcript is bit-exact with the golden model (edgecert.sumfold.prove) for logN <= --golden-max-logn; for larger logN the golden verifier is run on the kernel output and all configurations of the same logN must yield the identical transcript (cross-configuration consistency).  'contended' is true when any Vivado/xsim process was seen (Get-Process) before or after a logN block. The JSON is rewritten after every configuration, so a partial run is never lost.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernel_common as kc  # noqa: E402

FULL_LOGN = [12, 14, 16, 18, 20, 22, 24]
FULL_THREADS = [1, 8, 24, 32]
REDUCED_LOGN = [12, 16, 20, 24]
REDUCED_THREADS = [1, 8, 32]

def _ints(s: str):
    return [int(x) for x in s.split(",") if x.strip()]

def ratio_ci(a, b, n_boot=2000, seed=7):
    """Bootstrap 95% CI of median(a)/median(b)."""
    import random
    rng = random.Random(seed)
    rs = sorted(statistics.median(rng.choices(a, k=len(a))) / statistics.median(rng.choices(b, k=len(b)))
                for _ in range(n_boot))
    return [rs[int(0.025 * n_boot)], rs[int(0.975 * n_boot) - 1]]

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logn", type=_ints, default=None, help="comma list (default full grid)")
    ap.add_argument("--threads", type=_ints, default=None, help="comma list (default full grid)")
    ap.add_argument("--variants", default="fused,unfused")
    ap.add_argument("--compilers", default="gcc", help="comma list of gcc,msvc")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--sched", default="static", choices=["static", "dynamic"])
    ap.add_argument("--lanes", type=int, default=None, help="fused lanes override (default auto)")
    ap.add_argument("--pin", default="pfirst", choices=["pfirst", "none"],
                    help="pfirst: process affinity = one logical CPU per P-core, then E-cores, then "
                         "SMT siblings (first t of that order); none: OS placement")
    ap.add_argument("--seedf", type=lambda s: int(s, 16), default=kc.DEFAULT_SEED_F)
    ap.add_argument("--seedg", type=lambda s: int(s, 16), default=kc.DEFAULT_SEED_G)
    ap.add_argument("--golden-max-logn", type=int, default=16)
    ap.add_argument("--reduced", action="store_true", help=f"logN {REDUCED_LOGN}, threads {REDUCED_THREADS}")
    ap.add_argument("--auto-reduce", action="store_true", help="use the reduced grid if Vivado/xsim runs")
    ap.add_argument("--time-budget-s", type=float, default=1500.0,
                    help="stop starting new configurations after this many seconds")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--out", default=str(kc.ROOT / "results" / "raw" / "kernels" / "cpu_bench.json"))
    args = ap.parse_args(argv)

    t_start = time.time()
    snap0 = kc.contended_now()
    reduced = args.reduced or (args.auto_reduce and snap0["contended"])
    logns = args.logn or (REDUCED_LOGN if reduced else FULL_LOGN)
    threads = args.threads or (REDUCED_THREADS if reduced else FULL_THREADS)
    variants = [v for v in args.variants.split(",") if v]
    compilers = [c for c in args.compilers.split(",") if c]

    builds = {c: kc.build_cpu(force=args.rebuild, compiler=c) for c in compilers}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rec = {
        "status": "RUNNING",
        "evidence_class": "SUCCESS",
        "experiment_kind": "PROOF_KERNEL",
        "impl": "cpu_c",
        "kernel": "SumFold product sumcheck (spec/sumfold_kernel.md v1), Goldilocks + Ext2",
        "command": subprocess.list2cmdline([sys.executable] + sys.argv),
        "timestamp_utc_start": kc.now_utc(),
        "timestamp_utc_end": None,
        "host": kc.host_info(),
        "cpu_load_percent_at_start": kc.cpu_load_percent(),
        "builds": builds,
        "params": {"logn": logns, "threads": threads, "variants": variants, "compilers": compilers,
                   "warmup": args.warmup, "reps": args.reps, "sched": args.sched, "lanes": args.lanes,
                   "seed_f": f"{args.seedf:016x}", "seed_g": f"{args.seedg:016x}", "pin": args.pin,
                   "golden_max_logn": args.golden_max_logn, "reduced_grid": reduced,
                   "time_budget_s": args.time_budget_s},
        "timing_definition": ("total_s = wall time of sf_prove(): every pass, the SHA-256 transcript, "
                              "the lane gather and host tail; excludes test-vector generation and the "
                              "untimed re-expansion of the tables before each repetition. Timer: "
                              "QueryPerformanceCounter (100 ns ticks)."),
        "cpu_topology": [{"efficiency_class": c["efficiency_class"], "logical": c["logical"]}
                         for c in kc.cpu_topology()],
        "contended": bool(snap0["contended"]),
        "contention_definition": "contended = a Vivado/xsim process existed at any snapshot (before the sweep and after every logN block); active_cpu_pct = CPU used by those processes over 1 s, in % of all logical CPUs; host_busy = Win32_Processor LoadPercentage > 15 % while the runner itself is idle (any other load, e.g. other agents)",
        "contention_active_cpu_pct_max": snap0.get("active_cpu_pct"),
        "machine_load_pct_while_idle_max": snap0.get("machine_load_pct_while_idle"),
        "host_busy": bool(snap0.get("host_busy")),
        "contention_snapshots": [snap0],
        "results": [],
        "ablation_fused_vs_unfused": [],
        "consistency": {},
        "notes": [
            "Claim H is taken as s0+s1 of round 0 (identical field element to inner(f,g)); "
            "the golden prove() computes it with a separate pass.",
            "fused = spec 3.1 in-place fold-and-evaluate on lane-local contiguous blocks; "
            "unfused = evaluate pass + out-of-place ping-pong fold pass per round.",
            "i9-14900K is a hybrid CPU (8 P-cores with HT + 16 E-cores). With --pin pfirst the process "
            "affinity is the first t logical CPUs of [one per P-core, E-cores, P-core SMT siblings]; "
            "threads float inside that set (no per-thread binding).",
        ],
    }

    def flush(status=None):
        if status:
            rec["status"] = status
        rec["timestamp_utc_end"] = kc.now_utc()
        rec["elapsed_s"] = round(time.time() - t_start, 1)
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        tmp.replace(out_path)

    flush()
    stopped_early = False
    failures = 0
    for comp in compilers:
        exe = builds[comp]["exe"]
        for n in logns:
            if time.time() - t_start > args.time_budget_s:
                stopped_early = True
                break
            gold = kc.golden_record(args.seedf, args.seedg, n) if n <= args.golden_max_logn else None
            digests = set()
            for t in threads:
                mask = kc.affinity_mask(t, args.pin)
                for v in variants:
                    entry = {"compiler": comp, "variant": v, "n": n, "threads": t}
                    try:
                        r = kc.run_cpu(exe, n, t, v, args.seedf, args.seedg, reps=args.reps,
                                       warmup=args.warmup, lanes=args.lanes if v == "fused" else None,
                                       sched=args.sched if v == "fused" else None, affinity=mask)
                    except Exception as e:  # keep going, record the failure
                        entry.update({"status": "RUN_FAILED", "error": str(e)[:500]})
                        rec["results"].append(entry)
                        failures += 1
                        flush()
                        continue
                    tot = r["total_s"]
                    npass = len(r["pass_kinds"])
                    pass_med = [statistics.median(rep[p] for rep in r["pass_s"]) for p in range(npass)]
                    mism = kc.compare_records(r, gold) if gold is not None else None
                    dig = kc.transcript_digest(r)
                    digests.add(dig)
                    N = 1 << n
                    st = kc.summarize(tot)
                    entry.update({
                        "status": "SUCCESS" if (mism in (None, []) and r["consistent_across_runs"])
                        else "VERIFY_FAILED",
                        "exe_command": r["_command"],
                        "lanes": r["lanes"], "sched": r["sched"], "threads_used": r["threads_used"],
                        "affinity_mask": r.get("affinity_mask"),
                        "warmup": r["warmup"], "reps": r["reps"],
                        "total_s": tot, "total_stats_s": st,
                        "warmup_total_s": r["warmup_total_s"],
                        "transcript_median_s": statistics.median(r["transcript_s"]),
                        "tail_median_s": statistics.median(r["tail_s"]),
                        "pass_kinds": r["pass_kinds"], "pass_median_s": pass_med,
                        "elems_per_s_median": N / st["median"],
                        "ns_per_elem_median": st["median"] / N * 1e9,
                        "match": (mism == []) if gold is not None else None,
                        "match_basis": "golden edgecert.sumfold.prove" if gold is not None
                        else "golden verifier + cross-configuration consistency",
                        "mismatched_fields": mism,
                        "verify_ok": kc.verify_record(r),
                        "consistent_across_runs": r["consistent_across_runs"],
                        "transcript_sha256": dig,
                        "claim": r["claim"], "final_f": r["final_f"], "final_g": r["final_g"],
                    })
                    if entry["status"] != "SUCCESS" or not entry["verify_ok"]:
                        failures += 1
                    rec["results"].append(entry)
                    flush()
            snap = kc.contended_now()
            rec["contention_snapshots"].append(snap)
            rec["contended"] = rec["contended"] or bool(snap["contended"])
            acts = [x.get("active_cpu_pct") for x in rec["contention_snapshots"] if x.get("active_cpu_pct") is not None]
            rec["contention_active_cpu_pct_max"] = max(acts) if acts else None
            loads = [x.get("machine_load_pct_while_idle") for x in rec["contention_snapshots"]
                     if x.get("machine_load_pct_while_idle") is not None]
            rec["machine_load_pct_while_idle_max"] = max(loads) if loads else None
            rec["host_busy"] = any(x.get("host_busy") for x in rec["contention_snapshots"])
            key = f"{comp}:n={n}"
            rec["consistency"][key] = {"distinct_transcripts": len(digests), "all_equal": len(digests) == 1}
            if len(digests) > 1:  # configurations disagree: none of them can be trusted
                for e in rec["results"]:
                    if e["compiler"] == comp and e["n"] == n and e.get("status") == "SUCCESS":
                        e["status"] = "VERIFY_FAILED"
                        e["error"] = "transcript differs across configurations of the same logN"
                        failures += 1
            flush()
        if stopped_early:
            break

    # ---- ablation summary: fused vs unfused, same compiler / n / threads
    byk = {}
    for e in rec["results"]:
        if e.get("status") == "SUCCESS":
            byk[(e["compiler"], e["n"], e["threads"], e["variant"])] = e
    for (comp, n, t, v), e in sorted(byk.items()):
        if v != "fused" or (comp, n, t, "unfused") not in byk:
            continue
        u = byk[(comp, n, t, "unfused")]
        rec["ablation_fused_vs_unfused"].append({
            "compiler": comp, "n": n, "threads": t,
            "fused_median_s": e["total_stats_s"]["median"], "unfused_median_s": u["total_stats_s"]["median"],
            "speedup_unfused_over_fused": u["total_stats_s"]["median"] / e["total_stats_s"]["median"],
            "speedup_ci95_bootstrap": ratio_ci(u["total_s"], e["total_s"]),
        })
    # ---- thread scaling (vs 1 thread, same compiler / variant / n)
    scal = []
    for (comp, n, t, v), e in sorted(byk.items()):
        base = byk.get((comp, n, 1, v))
        if base is not None and t != 1:
            scal.append({"compiler": comp, "n": n, "variant": v, "threads": t,
                         "speedup_vs_1_thread": base["total_stats_s"]["median"] / e["total_stats_s"]["median"]})
    rec["thread_scaling"] = scal
    rec["cpu_load_percent_at_end"] = kc.cpu_load_percent()
    status = "SUCCESS" if failures == 0 else "PARTIAL_FAILURES"
    if stopped_early:
        rec["notes"].append(f"time budget {args.time_budget_s}s reached")
        status = "SUCCESS" if failures == 0 else "ABORT"
    flush(status)
    print(json.dumps({"out": str(out_path), "status": rec["status"], "configs": len(rec["results"]),
                      "contended": rec["contended"], "elapsed_s": rec["elapsed_s"]}))
    return 0 if failures == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
