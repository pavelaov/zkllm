"""CertSched discrete-event simulation experiments (C3).

  E3  load sweep: provisional / certified latency p50/p95/p99, throughput, max proof
      debt, backpressure events, aborts; 30 seeds; bootstrap CIs over seeds.
  E4  certified-latency CDF per policy at a fixed load.
  E2  proof-debt time series under a transient overload, with and without
      backpressure (debt bound B), 30 seeds (median / p10 / p90 bands).

Request classes: prefill in {32, 128, 512} x generated in {8, 32}, uniform. Scenarios (simulate.SCENARIOS): ws_4090, ws_split, edge16, edge8.

Outputs: results/raw/sched/{plans,des_load_sweep,des_debt_timeseries}.json
         results/processed/{sched_load_sweep,sched_latency_cdf,sched_debt_timeseries,sched_plans}.csv
Usage:   python bench/runners/sched_des.py [--seeds 30] [--jobs 6] [--reuse-plans] [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
from sched_common import GiB, RAW, inputs, meta, write_csv, write_json  # noqa: E402

import numpy as np  # noqa: E402

from edgecert.runtime.simulate import (SCENARIOS, SimConfig, arrivals, bootstrap_ci,  # noqa: E402
                                       inference_reserve, load_cost_model, request_plan,
                                       run_metrics, simulate)

MODELS = ["gpt2-124m", "qwen3-0.6b", "gemma3-1b"]
POLICIES = ["KEEP_ALL", "HOST_STAGE", "SPILL", "RECOMPUTE", "CERTSCHED"]
CLASSES = [(tp, tg) for tp in (32, 128, 512) for tg in (8, 32)]
LOADS = [0.3, 0.5, 0.7, 0.9, 1.1]
B_CERT = 128          # proof-debt bound (tokens) for the full CertSched configuration
N_REQ = 400

# variant name -> (memory policy, SimConfig overrides)
VARIANTS = {
    "KEEP_ALL": ("KEEP_ALL", {}),
    "HOST_STAGE": ("HOST_STAGE", {}),
    "SPILL": ("SPILL", {}),
    "RECOMPUTE": ("RECOMPUTE", {}),
    "CERTSCHED_MEM": ("CERTSCHED", {}),
    "CERTSCHED_FULL": ("CERTSCHED", {"B": B_CERT, "proof_order": "EDF",
                                     "shared_priority": "debt_aware"}),
}
METRICS = ["prov_p50", "prov_p95", "prov_p99", "cert_p50", "cert_p95", "cert_p99",
           "uncert_gap_p95", "throughput_tok_s", "throughput_req_s", "max_debt_tokens",
           "bp_events", "bp_fraction", "abort_rate", "deadline_miss_rate", "oom_events"]

def compute_plans(args):
    scenario, model = args
    cm = load_cost_model()
    out = []
    reserve = inference_reserve(model, CLASSES) if SCENARIOS[scenario]["shared"] else 0
    for pol in POLICIES:
        for tp, tg in CLASSES:
            p = request_plan(model, tp, tg, pol, scenario, cm, reserve_bytes=reserve)
            out.append(p)
    return out

def _cfg(scenario, cm, **kw):
    sc = SCENARIOS[scenario]
    unified = cm["devices"][sc["proof_dev"]].get("unified", False)
    return SimConfig(shared=sc["shared"], unified=bool(unified), cap=sc["cap_GiB"] * GiB,
                     host_cap=cm["host"]["app_budget_GiB"] * GiB,
                     ssd_cap=cm["storage"]["spill_budget_GB"] * 1e9, **kw)

def _with_sref(plans, ref_plans, shared):
    """Attach the class reference service time (KEEP_ALL on the same scenario)."""
    ref = {(p["Tp"], p["Tg"]): p for p in ref_plans}
    out = []
    for p in plans:
        r = ref[(p["Tp"], p["Tg"])]
        q = dict(p)
        q["_S_ref"] = (r["t_inf_s_est"] + r["t_proof_s_est"]) if shared else max(
            r["t_inf_s_est"], r["t_xfer_s_est"], r["t_proof_s_est"])
        out.append(q)
    return out

def des_job(args):
    scenario, model, plans_all, seeds, loads, cdf_load, series = args
    cm = load_cost_model()
    shared = SCENARIOS[scenario]["shared"]
    by_pol = {pol: [p for p in plans_all if p["policy"] == pol] for pol in POLICIES}
    ref = by_pol["KEEP_ALL"]
    mean_sref = float(np.mean([p["_S_ref"] for p in _with_sref(ref, ref, shared)]))
    summary, cdf_rows, ts = [], [], None
    for var, (pol, over) in VARIANTS.items():
        plans = _with_sref(by_pol[pol], ref, shared)
        for rho in loads:
            lam = rho / mean_sref
            per_seed = []
            cert_all = []
            for s in seeds:
                rng = random.Random(1_000_003 * s + 17)
                arr = arrivals(lambda t: lam, lam, N_REQ / lam, rng)
                out = simulate(plans, arr, _cfg(scenario, cm, **over), rng)
                m = run_metrics(out)
                per_seed.append(m)
                if abs(rho - cdf_load) < 1e-9:
                    n0 = int(len(out["reqs"]) * 0.1)
                    for r in out["reqs"][n0:]:
                        cert_all.append(math.inf if (r.aborted or math.isnan(r.cert)) else r.cert - r.arr)
            row = {"scenario": scenario, "model": model, "variant": var, "memory_policy": pol,
                   "B": over.get("B", "inf"), "proof_order": over.get("proof_order", "FIFO"),
                   "shared_priority": over.get("shared_priority", "infer_first"), "rho": rho,
                   "lambda_req_s_est": lam, "seeds": len(seeds)}
            for k in METRICS:
                vals = [m[k] for m in per_seed]
                mean, lo, hi = bootstrap_ci(vals)
                row[k + ("_est" if k not in ("bp_events", "bp_fraction", "abort_rate", "oom_events",
                                              "max_debt_tokens", "deadline_miss_rate") else "")] = mean
                row[k + "_ci_lo"] = lo
                row[k + "_ci_hi"] = hi
            row["plan_infeasible_classes"] = sum(1 for p in plans if not (p["legal"] and p["fits"]))
            summary.append(row)
            if cert_all:
                xs = np.sort(np.array(cert_all, dtype=float))
                fin = np.isfinite(xs)
                qs = np.arange(0, 101, 1)
                # nearest-rank quantiles; aborted / unfinished requests count as +inf
                qv = [xs[min(len(xs) - 1, max(0, int(math.ceil(q / 100.0 * len(xs))) - 1))] for q in qs]
                for q, v in zip(qs, qv):
                    cdf_rows.append({"scenario": scenario, "model": model, "variant": var, "rho": rho,
                                     "quantile": int(q), "cert_latency_s_est": float(v) if np.isfinite(v) else None,
                                     "n": int(xs.size), "completed_fraction": float(fin.mean())})
    if series:
        ts = debt_series(scenario, model, by_pol, ref, seeds, mean_sref, cm)
    return {"key": [scenario, model], "summary": summary, "cdf": cdf_rows, "series": ts}

def debt_series(scenario, model, by_pol, ref, seeds, mean_sref, cm):
    """Transient overload: rho 0.6 -> 1.5 -> 0.6. Debt / parked memory / admission queue."""
    shared = SCENARIOS[scenario]["shared"]
    base, peak = 0.6 / mean_sref, 1.5 / mean_sref
    H = 700 * mean_sref
    t1, t2 = 0.2 * H, 0.55 * H
    rate = lambda t: peak if t1 <= t < t2 else base   # noqa: E731
    dt = H / 350
    variants = {
        "KEEP_ALL_noBP": ("KEEP_ALL", {}),
        "CERTSCHED_noBP": ("CERTSCHED", {}),
        "CERTSCHED_B128": ("CERTSCHED", {"B": 128, "proof_order": "EDF", "shared_priority": "debt_aware"}),
        "CERTSCHED_B64": ("CERTSCHED", {"B": 64, "proof_order": "EDF", "shared_priority": "debt_aware"}),
    }
    out = {"scenario": scenario, "model": model, "horizon_s_est": H, "overload_window_s_est": [t1, t2],
           "rho_profile": {"base": 0.6, "overload": 1.5}, "dt_s_est": dt, "variants": {}}
    for name, (pol, over) in variants.items():
        plans = _with_sref(by_pol[pol], ref, shared)
        debt, mem, q = [], [], []
        per_seed = []
        for s in seeds:
            rng = random.Random(7_919 * s + 3)
            arr = arrivals(rate, peak, H, rng)
            res = simulate(plans, arr, _cfg(scenario, cm, **over), rng, series_dt=dt, horizon=H)
            sr = res["series"][:350]
            debt.append([x[1] for x in sr])
            mem.append([x[2] / GiB for x in sr])
            q.append([x[4] for x in sr])
            m = run_metrics(res, warmup_frac=0.0)
            per_seed.append({k: m[k] for k in ("max_debt_tokens", "bp_events", "oom_events", "abort_rate",
                                               "cert_p95", "cert_p99", "completed", "unfinished")})
        n = min(len(x) for x in debt)
        D = np.array([x[:n] for x in debt], dtype=float)
        M = np.array([x[:n] for x in mem], dtype=float)
        Q = np.array([x[:n] for x in q], dtype=float)
        out["variants"][name] = {
            "memory_policy": pol, "config": over,
            "t_s_est": [i * dt for i in range(n)],
            "debt_p10": np.quantile(D, 0.1, axis=0).tolist(), "debt_p50": np.quantile(D, 0.5, axis=0).tolist(),
            "debt_p90": np.quantile(D, 0.9, axis=0).tolist(),
            "parked_GiB_p50": np.quantile(M, 0.5, axis=0).tolist(),
            "admission_queue_p50": np.quantile(Q, 0.5, axis=0).tolist(),
            "per_seed": per_seed,
            "max_debt_over_seeds": float(D.max()),
        }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--reuse-plans", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cmd = (f"python bench/runners/sched_des.py --seeds {a.seeds} --jobs {a.jobs}"
           + (" --reuse-plans" if a.reuse_plans else "") + (" --quick" if a.quick else ""))
    cm = load_cost_model()
    est = inputs(cm)
    models = MODELS if not a.quick else ["gpt2-124m"]
    scen = list(SCENARIOS) if not a.quick else ["ws_4090", "edge16"]
    pairs = [(s, m) for s in scen for m in models]
    t0 = time.time()
    plan_path = os.path.join(RAW, "plans.json")
    plans = None
    if a.reuse_plans and os.path.exists(plan_path):
        with open(plan_path, encoding="utf-8") as f:
            doc = json.load(f)
        plans = [p for p in doc["plans"] if (p["scenario"], p["model"]) in set(pairs)]
        if len({(p["scenario"], p["model"]) for p in plans}) < len(pairs):
            plans = None
    if plans is None:
        plans = []
        with Pool(a.jobs) as pool:
            for out in pool.imap_unordered(compute_plans, pairs):
                plans += out
                print("plans", out[0]["scenario"], out[0]["model"], f"{time.time() - t0:.0f}s", flush=True)
        write_json("plans.json", {
            "status": "SIMULATION_ONLY", "evidence_class": "SIMULATION", **meta(cmd, cm),
            "plans": plans})
        write_csv("sched_plans.csv", plans, [
            "scenario", "model", "policy", "Tp", "Tg", "legal", "fits", "selected", "reserve_bytes",
            "peak_device_bytes", "peak_host_bytes", "peak_ssd_bytes", "witness_bytes", "inf_ws_bytes",
            "t_inf_s_est", "t_xfer_s_est", "t_proof_s_est", "t_overhead_s_est", "t_verify_s_est"])
    seeds = list(range(a.seeds))
    loads = LOADS if not a.quick else [0.5, 0.9]
    jobs = []
    for s, m in pairs:
        pp = [p for p in plans if p["scenario"] == s and p["model"] == m]
        jobs.append((s, m, pp, seeds, loads, 0.7 if not a.quick else 0.5,
                     (s, m) in (("ws_4090", "qwen3-0.6b"), ("edge16", "gpt2-124m"), ("ws_4090", "gpt2-124m"))))
    summary, cdf, series = [], [], []
    with Pool(a.jobs) as pool:
        for out in pool.imap_unordered(des_job, jobs):
            summary += out["summary"]
            cdf += out["cdf"]
            if out["series"]:
                series.append(out["series"])
            print("des", out["key"], f"{time.time() - t0:.0f}s", flush=True)
    summary.sort(key=lambda r: (r["scenario"], r["model"], r["variant"], r["rho"]))
    write_json("des_load_sweep.json", {
        "status": "SIMULATION_ONLY", "evidence_class": "SIMULATION", **meta(cmd, cm),
        "inputs": est,
        "design": {"classes": CLASSES, "loads_rho": loads, "requests_per_run": N_REQ, "warmup_fraction": 0.1,
                   "seeds": seeds, "bootstrap": "percentile, 2000 resamples over per-seed statistics",
                   "rho_definition": "lambda * mean class service time of KEEP_ALL on the bottleneck stage",
                   "variants": {k: {"memory_policy": v[0], **v[1]} for k, v in VARIANTS.items()},
                   "release": "strict release: user-visible latency = certified; preview: provisional"},
                   "rows": summary, "elapsed_s": time.time() - t0})
    write_csv("sched_load_sweep.csv", summary)
    write_csv("sched_latency_cdf.csv", cdf)
    write_json("des_debt_timeseries.json", {
        "status": "SIMULATION_ONLY", "evidence_class": "SIMULATION", **meta(cmd, cm),
        "inputs": est, "series": series})
    ts_rows = []
    for s in series:
        for vname, v in s["variants"].items():
            for i, t in enumerate(v["t_s_est"]):
                ts_rows.append({"scenario": s["scenario"], "model": s["model"], "variant": vname,
                                "t_s_est": t, "debt_p10": v["debt_p10"][i], "debt_p50": v["debt_p50"][i],
                                "debt_p90": v["debt_p90"][i], "parked_GiB_p50": v["parked_GiB_p50"][i],
                                "admission_queue_p50": v["admission_queue_p50"][i],
                                "overload_start_s_est": s["overload_window_s_est"][0],
                                "overload_end_s_est": s["overload_window_s_est"][1]})
    write_csv("sched_debt_timeseries.csv", ts_rows)
    print("done", f"{time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
