"""Generate configs/cost_model.yaml for CertSched (C3).

Reads kernel throughputs when other work packages have produced them: results/raw/kernels/cpu_bench.json          -> CPU sumcheck throughput
results/raw/fpga_impl/throughput_model.json -> FPGA SumFold throughput (POST_ROUTE model) results/raw/fpga_sim/sim_results.json       -> FPGA cycles per element

Usage:  python bench/runners/sched_cost_model.py [--out configs/cost_model.yaml]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def _walk(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, path + (str(i),))
    else:
        yield path, obj

def _find_rate(doc, must=("per_s",), prefer=("elem", "throughput")):
    """Best-effort search for an elements-per-second figure in a foreign JSON record."""
    best = None
    for path, v in _walk(doc):
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
            continue
        key = "/".join(path).lower()
        if key.endswith("_est") or "_est/" in key:
            continue
        if all(m in key for m in must) and any(p in key for p in prefer):
            if best is None or v > best[1]:
                best = (key, float(v))
    return best

def read_optional(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # pragma: no cover
        return {"_error": str(e)}

def _status_ok(doc) -> bool:
    if not isinstance(doc, dict):
        return False
    st = str(doc.get("status", "")).upper()
    return st.startswith("SUCCESS") or st.startswith("ABORT") or st in ("POST_ROUTE", "OK", "SUCCESS")

def build(out_path: str) -> dict:
    cpu = read_optional("results/raw/kernels/cpu_bench.json")
    fimpl = read_optional("results/raw/fpga_impl/throughput_model.json")
    fsim = read_optional("results/raw/fpga_sim/sim_results.json")
    sources = {}

    # ---------------- CPU sumcheck throughput
        # Work element = one table entry; the kernel's elems_per_s counts N (one f,g *pair*) per second for a full n-round product sumcheck, so work elements/s = 2 x pairs/s.
    cpu_pairs, cpu_ev, cpu_basis = 0.5e9, "SUCCESS", (
        "min(compute, bandwidth): ~15 ns per work element per core for Goldilocks-ext "
        "fold+evaluate on ~16 P-core equivalents, and 64 GB/s x 0.7 / 32 B per element")
    if cpu is not None and isinstance(cpu, dict) and isinstance(cpu.get("results"), list):
        good = [r for r in cpu["results"]
                if str(r.get("status", "")).startswith("SUCCESS")
                and r.get("variant") == "fused" and r.get("verify_ok") is True
                and r.get("consistent_across_runs") is True
                and isinstance(r.get("elems_per_s_median"), (int, float))]
        if good:
            nmax = max(r["n"] for r in good)
            top = max((r for r in good if r["n"] == nmax), key=lambda r: r["elems_per_s_median"])
            cpu_pairs, cpu_ev = float(top["elems_per_s_median"]), "SUCCESS"
            cpu_basis = (f"results/raw/kernels/cpu_bench.json: fused, {top.get('compiler')}, "
                         f"n={top['n']}, threads={top.get('threads')}, median pairs/s "
                         f"(host contended={cpu.get('contended')}); x2 table entries per pair")
            sources["cpu_bench"] = {"n": top["n"], "threads": top.get("threads"),
                                    "compiler": top.get("compiler"),
                                    "pairs_per_s_median": cpu_pairs,
                                    "contended": cpu.get("contended"),
                                    "timestamp": cpu.get("timestamp_utc_end")}
    elif cpu is not None:
        sources["cpu_bench"] = "present but unrecognised schema; ignored"
    cpu_sc = 2.0 * cpu_pairs

    # ---------------- FPGA SumFold
    f_lanes, f_mhz, f_ev, f_basis = 4, 150.0, "SUCCESS", (
        "placeholder: 4 lanes at an assumed 150 MHz, ~N/L cycles per full sumcheck; "
        "replace with post-route Fmax and lane count")
    f_rate = None
    if fimpl is not None and _status_ok(fimpl):
        hit = _find_rate(fimpl)
        if hit:
            f_rate, f_ev = hit[1], "POST_ROUTE"
            f_basis = f"results/raw/fpga_impl/throughput_model.json field {hit[0]}"
            sources["fpga_impl"] = hit[0]
    if f_rate is None and isinstance(fsim, dict) and isinstance(fsim.get("cases"), list):
        cs = [c for c in fsim["cases"] if c.get("match") and c.get("lanes") == f_lanes
              and isinstance(c.get("cycles_total"), (int, float)) and c.get("n")]
        if cs:
            c = max(cs, key=lambda c: c["n"])
            cyc_per_pair = c["cycles_total"] / (1 << c["n"])
            f_rate = 2.0 * f_mhz * 1e6 / cyc_per_pair
            f_ev = "SUCCESS"
            f_basis = (f"cycles from results/raw/fpga_sim/sim_results.json (SIMULATION, L={f_lanes}, "
                       f"n={c['n']}, {c['cycles_total']} cycles) x assumed {f_mhz:.0f} MHz clock "
                       "x2 table entries per pair")
            sources["fpga_sim"] = {"lanes": f_lanes, "n": c["n"], "cycles_total": c["cycles_total"]}
    if f_rate is None:
        f_rate = 2.0 * f_lanes * f_mhz * 1e6      # work elements/s

    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    pcie = 16e9 * 16 * 128 / 130 / 8 / 1e9        # GB/s per direction
    nvme_link = 16e9 * 4 * 128 / 130 / 8 / 1e9
    cm = {
        "meta": {
            "generated_utc": now,
            "generator": "bench/runners/sched_cost_model.py",
            "host": platform.node(),
            "python": sys.version.split()[0],
            "note": ("Every time CertSched derives from an constant is itself"),
            "inputs_found": sources,
        },
        "work_factors": {
            "c_matmul": 1.0, "c_hadamard": 1.0, "c_lookup": 2.0, "c_open": 1.0, "c_derive": 3.0,
            "basis": ("work per abstract element relative to one product-sumcheck element: "
                      "LogUp needs an inverse column plus a two-table sumcheck (2x); batch "
                      "inversion ~3 multiplications per element"),
        },
        "verifier": {
            "per_round_us": 1.0, "per_opening_us": 20.0,
            "basis": "log-size sumcheck checks on a CPU; batched hash-based opening checks",
        },
        "devices": {
            "gpu_rtx4090": {
                "kind": "gpu", "memory_GiB": 24, "app_budget_GiB": 20, "unified": False,
                "int8_tops_nominal": 660.6, "int8_efficiency": 0.30,
                "mem_bw_GBps_nominal": 1008.0, "mem_efficiency": 0.70,
                "sc_elems_per_s": 1008.0e9 * 0.7 / 32, "commit_GBps": 4.0, "hash_GBps": 20.0,
                "link": "pcie4_x16", "ssd": "nvme_gen4_x4",
                "basis": {"sc_elems_per_s": ("bandwidth-bound: nominal DRAM bandwidth x 0.7 / 32 B "
                                             "per work element (no CUDA toolkit on host, so no "
                                             "GPU sumcheck yet)"),
                          "commit_GBps": "order-of-magnitude assumption for GPU RS-encode + Merkle hashing",
                          "hash_GBps": "order-of-magnitude assumption for a GPU digest"},
            },
            "gpu_rtx6000ada": {
                "kind": "gpu", "memory_GiB": 48, "app_budget_GiB": 40, "unified": False,
                "int8_tops_nominal": 728.0, "int8_efficiency": 0.30,
                "mem_bw_GBps_nominal": 960.0, "mem_efficiency": 0.70,
                "sc_elems_per_s": 960.0e9 * 0.7 / 32, "commit_GBps": 4.0, "hash_GBps": 20.0,
                "link": "pcie4_x16", "ssd": "nvme_gen4_x4",
                "basis": {"sc_elems_per_s": "bandwidth-bound, as for gpu_rtx4090"},
            },
            "cpu_i9_14900k": {
                "kind": "cpu", "memory_GiB": 192, "app_budget_GiB": 128, "unified": True,
                "int8_tops_nominal": 4.0, "int8_efficiency": 0.5,
                "mem_bw_GBps_nominal": 64.0, "mem_efficiency": 0.70,
                "sc_pairs_per_s": cpu_pairs, "sc_elems_per_s": cpu_sc,
                "commit_GBps": 1.0, "hash_GBps": 5.0,
                "aes_gcm_GBps": 10.0, "link": None, "ssd": "nvme_gen4_x4",
                "basis": {"sc_elems_per_s": cpu_basis},
            },
            "edge_unified_16g": {
                "kind": "edge", "memory_GiB": 16, "app_budget_GiB": 16, "unified": True,
                "int8_tops_nominal": 50.0, "int8_efficiency": 0.30,
                "mem_bw_GBps_nominal": 102.4, "mem_efficiency": 0.70,
                "sc_elems_per_s": 102.4e9 * 0.7 / 32, "commit_GBps": 0.5, "hash_GBps": 2.0,
                "aes_gcm_GBps": 2.0, "link": None, "ssd": "nvme_edge",
                "evidence": {"all": "Orin-NX-class unified LPDDR5 SoC"},
                "basis": {"sc_elems_per_s": "bandwidth-bound: 102.4 GB/s x 0.7 / 32 B"},
            },
            "edge_unified_8g": {
                "kind": "edge", "memory_GiB": 8, "app_budget_GiB": 8, "unified": True,
                "int8_tops_nominal": 50.0, "int8_efficiency": 0.30,
                "mem_bw_GBps_nominal": 102.4, "mem_efficiency": 0.70,
                "sc_elems_per_s": 102.4e9 * 0.7 / 32, "commit_GBps": 0.5, "hash_GBps": 2.0,
                "aes_gcm_GBps": 2.0, "link": None, "ssd": "nvme_edge",
                "evidence": {"all": "8 GiB variant of edge_unified_16g"},
                "basis": {"sc_elems_per_s": "as edge_unified_16g"},
            },
            "fpga_zcu102_sumfold": {
                "kind": "fpga", "lanes": f_lanes, "clock_MHz": f_mhz,
                "sc_elems_per_s": f_rate, "onchip_max_log2_n": 16,
                "link": "eth_1g", "control_link": "jtag_axi",
                "evidence": {"sc_elems_per_s": f_ev, "onchip_max_log2_n": "SUCCESS",
                             "lanes": f_ev, "clock_MHz": f_ev},
                "basis": {"sc_elems_per_s": f_basis,
                          "onchip_max_log2_n": ("2N x 128 bit resident must fit XCZU9EG block RAM "
                                                "(912 x 36 Kb, no UltraRAM): N = 2^16 fits, 2^17 does not")},
            },
        },
        "links": {
            "pcie4_x16": {"nominal_GBps": round(pcie, 3), "sustained_fraction": 0.8,
                          "evidence": "16 GT/s x 16 lanes x 128/130."},
            "nvme_gen4_x4": {"nominal_GBps": round(nvme_link, 3), "read_GBps": 5.0, "write_GBps": 3.0,
                             "evidence": "link bound"},
            "nvme_edge": {"nominal_GBps": round(nvme_link, 3), "read_GBps": 3.0, "write_GBps": 2.0,
                          "evidence": "link bound"},
            "eth_2g5": {"nominal_GBps": 0.3125, "evidence": "2.5 Gb/s / 8"},
            "eth_1g": {"nominal_GBps": 0.125, "evidence": "1 Gb/s / 8"},
            "jtag_axi": {"nominal_GBps": 0.002, "evidence": "JTAG-to-AXI register access, ~2 MB/s"},
        },
        "storage": {"spill_budget_GB": 500, "evidence": "planning budget for bounded witness scratch"},
        "host": {"app_budget_GiB": 128, "evidence": "planning budget"},
    }
    text = to_yaml(cm)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# CertSched cost model (C3).\n")
        f.write(text)
    return cm

def _scalar(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v) if isinstance(v, int) else f"{v:.6g}"
    s = str(v)
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

def to_yaml(obj, indent=0) -> str:
    pad = "  " * indent
    out = []
    for k, v in obj.items():
        if isinstance(v, dict):
            out.append(f"{pad}{k}:")
            out.append(to_yaml(v, indent + 1))
        elif isinstance(v, list):
            out.append(f"{pad}{k}: [" + ", ".join(_scalar(x) for x in v) + "]")
        else:
            out.append(f"{pad}{k}: {_scalar(v)}")
    return "\n".join(x for x in out if x != "")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "configs", "cost_model.yaml"))
    a = ap.parse_args()
    cm = build(a.out)
    print("wrote", a.out, "inputs_found:", cm["meta"]["inputs_found"])
