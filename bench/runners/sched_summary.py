"""Headline numbers for the CertSched (C3) paper text, extracted from results/raw/sched.

Every value keeps its evidence class; values derived from ESTIMATE constants are in fields ending in _est (render with \\est{} in LaTeX).

Output: results/raw/sched/summary.json Usage:  python bench/runners/sched_summary.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
from sched_common import RAW, meta, write_json  # noqa: E402

def load(name):
    with open(os.path.join(RAW, name), encoding="utf-8") as f:
        return json.load(f)

def main():
    out = {"status": "SIMULATION_ONLY", "evidence_class": "SIMULATION",
           **meta("python bench/runners/sched_summary.py")}
    # ---------------- legality
    leg = load("legality.json")["rows"]
    out["legality"] = {
        "evidence": "SIMULATION (exact replay, no cost constants)",
        "cases": len(leg), "all_correct": all(r["correct"] for r in leg),
        "policies_accepted": sorted({r["case"] for r in leg if r["kind"] == "policy" and r["legal"]}),
        "policies_rejected": sorted({r["case"] for r in leg if r["kind"] == "policy" and not r["legal"]}),
        "mutations_rejected": sorted({r["case"] for r in leg if r["kind"] == "mutation" and not r["legal"]}),
        "mutations_accepted": sorted({r["case"] for r in leg if r["kind"] == "mutation" and r["legal"]}),
    }
    # ---------------- memory
    mem = load("memory_vs_ctx.json")["rows"]
    peaks = {}
    for r in mem:
        if r["policy"] in ("KEEP_ALL", "LAYER_STREAM_NAIVE") and r["cap_GiB"] is None:
            peaks.setdefault(r["protocol"], {}).setdefault(r["model"], {}).setdefault(r["policy"], {})[
                str(r["ctx"])] = round(r["peak_device_GiB"], 3)
    out["peak_memory_GiB"] = {"evidence": "ANALYTIC byte accounting of replayed schedules (stated encoding)",
                              "by_protocol_model_policy_ctx": peaks}
    cells = {}
    for r in mem:
        if r["protocol"] != "GLOBAL" or r["cap_GiB"] is None:
            continue
        k = (r["model"], r["ctx"], r["cap_GiB"])
        cells.setdefault(k, {})[r["policy"]] = r
    fit_cs = fit_uni = only_cs = 0
    ratios = []
    table = []
    for (m, c, cap), d in sorted(cells.items()):
        cs = d.get("CERTSCHED")
        uni = [d[p] for p in ("KEEP_ALL", "HOST_STAGE", "SPILL", "RECOMPUTE") if p in d and d[p].get("fits_cap")]
        best = min(uni, key=lambda r: r["added_s_est"]) if uni else None
        cs_fit = bool(cs and cs.get("fits_cap"))
        fit_cs += cs_fit
        fit_uni += best is not None
        only_cs += cs_fit and best is None
        entry = {"model": m, "ctx": c, "cap_GiB": cap, "certsched_fits": cs_fit,
                 "certsched_selected": cs.get("selected") if cs else None,
                 "certsched_added_s_est": cs["added_s_est"] if cs_fit else None,
                 "best_uniform": best["policy"] if best else None,
                 "best_uniform_added_s_est": best["added_s_est"] if best else None,
                 "keep_all_fits": bool(d.get("KEEP_ALL", {}).get("fits_cap"))}
        if cs_fit and best is not None and best["added_s_est"] > 1e-9:
            entry["added_time_ratio_est"] = cs["added_s_est"] / best["added_s_est"]
            if not entry["keep_all_fits"]:
                ratios.append(entry["added_time_ratio_est"])
        table.append(entry)
    out["certsched_vs_uniform"] = {
        "evidence": "fits: SIMULATION; added times: ESTIMATE (_est)",
        "cells": len(cells), "certsched_fits": fit_cs, "some_uniform_fits": fit_uni,
        "only_certsched_fits": only_cs,
        "added_time_ratio_vs_best_uniform_est": {
            "n": len(ratios), "median": statistics.median(ratios) if ratios else None,
            "min": min(ratios) if ratios else None, "max": max(ratios) if ratios else None,
            "note": "cells where KEEP_ALL does not fit and both CertSched and a uniform policy fit"},
        "table": table}
    # ---------------- optimality gap
    gap = load("optimality_gap.json")
    syn = gap["synthetic"]
    out["optimality_gap"] = {
        "synthetic_evidence": "SIMULATION (algorithmic)",
        "synthetic_mean_gap": statistics.mean(v["certsched_greedy"]["mean"] for v in syn.values()),
        "synthetic_max_gap": max(v["certsched_greedy"]["max"] for v in syn.values()),
        "synthetic_min_frac_optimal": min(v["certsched_greedy"]["frac_optimal"] for v in syn.values()),
        "synthetic_ratio_only_max_gap": max(v["ratio_greedy"]["max"] for v in syn.values()),
        "model_small_gap_est": gap["model_small_gap_summary_est"],
        "model_full_gap_vs_lp_upper_bound_est": {r["regime"]: r["gap_vs_lp_upper_bound"] for r in gap["model_full"]},
    }
    # ---------------- DES
    des = load("des_load_sweep.json")["rows"]

    def pick(sc, m, var, rho):
        for r in des:
            if r["scenario"] == sc and r["model"] == m and r["variant"] == var and abs(r["rho"] - rho) < 1e-9:
                return r
        return None

    keys = ["cert_p50_est", "cert_p95_est", "cert_p99_est", "prov_p95_est", "throughput_tok_s_est",
            "max_debt_tokens", "abort_rate", "bp_events", "oom_events"]
    snap = {}
    for sc in ("ws_4090", "ws_split", "edge16", "edge8"):
        for m in ("gpt2-124m", "qwen3-0.6b", "gemma3-1b"):
            for rho in (0.7, 0.9):
                for var in ("KEEP_ALL", "HOST_STAGE", "SPILL", "RECOMPUTE", "CERTSCHED_MEM", "CERTSCHED_FULL"):
                    r = pick(sc, m, var, rho)
                    if r:
                        snap[f"{sc}/{m}/rho{rho}/{var}"] = {k: r.get(k) for k in keys} | {
                            "cert_p95_ci": [r.get("cert_p95_ci_lo"), r.get("cert_p95_ci_hi")]}
    full_debt = [r["max_debt_tokens"] for r in des if r["variant"] == "CERTSCHED_FULL"]
    out["des"] = {"evidence": "SIMULATION; times ESTIMATE (_est); counts SIMULATION",
                  "certsched_full_max_debt_over_all_configs": max(full_debt) if full_debt else None,
                  "debt_bound_B": 128, "snapshots": snap}
    ts = load("des_debt_timeseries.json")["series"]
    out["overload"] = {}
    for s in ts:
        for name, v in s["variants"].items():
            ps = v["per_seed"]
            out["overload"][f"{s['scenario']}/{s['model']}/{name}"] = {
                "max_debt_over_seeds": v["max_debt_over_seeds"],
                "peak_median_debt": max(v["debt_p50"]),
                "peak_median_parked_GiB": max(v["parked_GiB_p50"]),
                "mean_oom_aborts_per_run": statistics.mean(p["oom_events"] for p in ps),
                "mean_abort_rate": statistics.mean(p["abort_rate"] for p in ps),
                "mean_bp_events": statistics.mean(p["bp_events"] for p in ps),
                "mean_cert_p99_s_est": statistics.mean(p["cert_p99"] for p in ps if p["cert_p99"] is not None),
            }
    oa = load("offload_admission.json")["rows"]
    out["offload_admission_est"] = [{k: r[k] for k in ("link", "local_device", "eligible_tasks", "admitted_tasks",
                                                        "best_T_offload_over_T_local_est")} for r in oa]
    p = write_json("summary.json", out)
    print("wrote", p)

if __name__ == "__main__":
    main()
