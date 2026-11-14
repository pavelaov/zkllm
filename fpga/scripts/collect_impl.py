"""collect_impl.py : parse the Vivado implementation runs in fpga/build/ and write the result records.

usage (repository root):  python fpga/scripts/collect_impl.py

Reads   fpga/build/L<L>_<MHz>MHz[_<strategy>]/{summary.txt, vivado.log, reports/*.rpt, *.bit, *.ltx}
        results/raw/fpga_sim/sim_results.json (simulated cycle counts, n = 16 AUTO cases)
Writes  results/raw/fpga_impl/runs.json            (list of run records)
        results/raw/fpga_impl/throughput_model.json
        results/processed/fpga_impl_summary.csv
        fpga/reports/<run>/                        (copies of the text reports + summary + log)
        fpga/build/RECOMMENDED.txt
"""
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
BUILD = os.path.join(REPO, "fpga", "build")
REPORTS = os.path.join(REPO, "fpga", "reports")
RAW = os.path.join(REPO, "results", "raw", "fpga_impl")
PROC = os.path.join(REPO, "results", "processed")
SIM = os.path.join(REPO, "results", "raw", "fpga_sim", "sim_results.json")
VIVADO = r"<vivado-2022.2>\bin\vivado.bat"

DEVICE = {"lut": 274080, "ff": 548160, "bram36_tiles": 912, "dsp": 2520}
MARGIN_NS = 0.05

def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")

def rel(p):
    return os.path.relpath(p, REPO).replace("\\", "/") if p else None

def read(p):
    try:
        with open(p, "r", errors="replace") as f:
            return f.read()
    except OSError:
        return ""

def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()

def parse_summary(p):
    kv = {}
    for line in read(p).splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv

# ------------------------------------------------------------------ report parsers
def section(txt, start, end):
    i = txt.find(start)
    if i < 0:
        return ""
    j = txt.find(end, i + len(start))
    return txt[i:j if j > 0 else len(txt)]

def parse_timing_summary(txt):
    """Design Timing Summary + per-clock (Intra Clock) table of report_timing_summary."""
    out = {"design": None, "clocks": {}}
    lines = txt.splitlines()
    for i, ln in enumerate(lines):
        if "WNS(ns)" in ln and "TNS(ns)" in ln and "WHS(ns)" in ln and out["design"] is None:
            # header, dashes, values
            for v in lines[i + 2:i + 4]:
                parts = v.split()
                if len(parts) >= 12 and fnum(parts[0]) is not None:
                    keys = ["wns_ns", "tns_ns", "tns_failing_endpoints", "tns_total_endpoints",
                            "whs_ns", "ths_ns", "ths_failing_endpoints", "ths_total_endpoints",
                            "wpws_ns", "tpws_ns", "tpws_failing_endpoints", "tpws_total_endpoints"]
                    out["design"] = {k: (fnum(x) if "ns" in k else int(float(x))) for k, x in zip(keys, parts)}
                    break
    # Intra Clock Table
    sec = section(txt, "| Intra Clock Table", "| Inter Clock Table")
    if sec:
        for ln in sec.splitlines():
            parts = ln.split()
            if len(parts) >= 13 and fnum(parts[1]) is not None:
                out["clocks"][parts[0]] = {
                    "wns_ns": fnum(parts[1]), "tns_ns": fnum(parts[2]),
                    "tns_failing_endpoints": int(parts[3]), "tns_total_endpoints": int(parts[4]),
                    "whs_ns": fnum(parts[5]), "ths_ns": fnum(parts[6]),
                    "ths_failing_endpoints": int(parts[7]), "ths_total_endpoints": int(parts[8]),
                    "wpws_ns": fnum(parts[9])}
    # Clock Summary (name, waveform, period, frequency)
    sec = section(txt, "| Clock Summary", "| Intra Clock Table")
    out["clock_summary"] = {}
    if sec:
        for ln in sec.splitlines():
            mm = re.match(r"^\s*(\S+)\s+\{([^}]*)\}\s+([\d.]+)\s+([\d.]+)", ln)
            if mm:
                out["clock_summary"][mm.group(1)] = {"period_ns": float(mm.group(3)),
                                                     "frequency_mhz": float(mm.group(4))}
    return out

def util_row(txt, name):
    """First table row whose first cell equals name: returns (used, available, util%)."""
    for ln in txt.splitlines():
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) >= 5 and cells[0] == name:
            nums = [fnum(c) for c in cells[1:]]
            used = nums[0]
            avail = nums[-2]
            pct = nums[-1]
            return used, avail, pct
    return None, None, None

def parse_util(txt):
    rows = {
        "lut": "CLB LUTs", "lut_logic": "LUT as Logic", "lut_mem": "LUT as Memory",
        "ff": "CLB Registers", "carry8": "CARRY8", "clb": "CLB",
        "bram36_tiles": "Block RAM Tile", "ramb36": "RAMB36/FIFO*", "ramb18": "RAMB18",
        "dsp": "DSPs", "dsp48e2": "DSP48E2 only", "bufgce": "BUFGCE", "mmcm": "MMCM",
        "bscane2": "BSCANE2",
    }
    out = {}
    for k, name in rows.items():
        used, avail, pct = util_row(txt, name)
        if used is None and name.endswith("*"):
            used, avail, pct = util_row(txt, name.rstrip("*"))
        out[k] = {"used": used, "available": avail, "util_pct": pct}
    return out

def parse_power(txt):
    def val(label):
        m = re.search(r"\|\s*" + re.escape(label) + r"\s*\|\s*([^|]+?)\s*\|", txt)
        return m.group(1).strip() if m else None
    out = {
        "total_w": fnum(val("Total On-Chip Power (W)")),
        "dynamic_w": fnum(val("Dynamic (W)")),
        "static_w": fnum(val("Device Static (W)")),
        "junction_temp_c": fnum(val("Junction Temperature (C)")),
        "confidence_level": val("Confidence Level"),
        "effective_theta_ja": val("Effective TJA (C/W)"),
    }
    comp = {}
    m = re.search(r"1\. Summary.*?On-Chip Components\n-+\n\n(.*?)\n\n", txt, re.S)
    if not m:
        m = re.search(r"On-Chip Components\n-+\n\n(.*?)\n\n", txt, re.S)
    if m:
        for ln in m.group(1).splitlines():
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if len(cells) >= 2 and fnum(cells[1]) is not None and cells[0] not in ("On-Chip",):
                comp[cells[0]] = fnum(cells[1])
    out["components_w"] = comp
    return out

def parse_rule_table(txt):
    """report_drc / report_methodology summary table: | Rule | Severity | Description | Violations |"""
    rules = []
    for ln in txt.splitlines():
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[1] in ("Error", "Critical Warning", "Warning", "Advisory"):
            try:
                n = int(cells[3])
            except ValueError:
                continue
            rules.append({"rule": cells[0], "severity": cells[1], "description": cells[2], "count": n})
    by = {}
    for r in rules:
        by[r["severity"]] = by.get(r["severity"], 0) + r["count"]
    m = re.search(r"Checks found:\s*(\d+)", txt)
    return {"checks_found": int(m.group(1)) if m else None, "by_severity": by, "rules": rules}

def parse_route_status(txt):
    out = {}
    for key, label in [("routable_nets", "# of routable nets"),
                       ("fully_routed_nets", "# of fully routed nets"),
                       ("nets_with_routing_errors", "# of nets with routing errors")]:
        m = re.search(re.escape(label) + r"\.*\s*:\s*(\d+)", txt)
        out[key] = int(m.group(1)) if m else None
    return out

def log_messages(log):
    cw = [ln for ln in log.splitlines() if ln.startswith("CRITICAL WARNING")]
    er = [ln for ln in log.splitlines() if ln.startswith("ERROR")]
    uniq = {}
    for ln in cw:
        m = re.match(r"CRITICAL WARNING: \[([^\]]+)\]", ln)
        k = m.group(1) if m else ln[:60]
        uniq.setdefault(k, ln[:300])
    return {"critical_warning_lines": len(cw), "critical_warning_ids": uniq,
            "error_lines": er[:20]}

# ------------------------------------------------------------------ one run
RTL_VERSIONS = {
    "v1_inferred_mul": {"gl_mul": "behavioural a*b + 6 registers, pipelined into a 16-DSP48E2 cascade by Vivado",
           "board_note": "mis-synthesized: two partial products one cycle early/late (netlist impulse test); "
                         "wrong round messages on the board (docs/notes/fpga_board.md)"},
    "v2_explicit_mul": {"gl_mul": "explicit 12-DSP48E2 decomposition, every pipeline register in RTL (fix of 2026-09-27)",
                        "board_note": "netlist-checked (fpga/sim/netlist) before the board run; bit-exact on the "
                                      "board (results/raw/fpga_board/campaign.json)"},
}
V1_BOARD_STATUS = "FUNCTIONALLY_INCORRECT_ON_BOARD"
V1_BOARD_BASIS = ("Wrong results on the board for v1 L8 250 MHz, L8 300 MHz perf and L1 250 MHz "
                  "(results/raw/fpga_board/bringup_L8_250_auto_n10.json, v1_L8_300_perf_auto_n10.json, "
                  "v1_L1_250_auto_n10.json); the other v1 configurations share the mis-synthesized gl_mul "
                  "(netlist SIMULATION, results/raw/fpga_sim/netlist_gl_mul.json)")
BOARD_DIR = os.path.join(REPO, "results", "raw", "fpga_board")

def board_records():
    """On-board test records, keyed by bitstream SHA-256 or by bitstream path."""
    out = []
    files = (glob.glob(os.path.join(BOARD_DIR, "*.json")) + glob.glob(os.path.join(BOARD_DIR, "probes", "*.json"))
             + [f for f in glob.glob(os.path.join(BOARD_DIR, "campaign", "*.json"))
                if not os.path.basename(f).startswith("_")])
    for f in sorted(files):
        try:
            r = json.load(open(f))
        except (OSError, ValueError):
            continue
        if not isinstance(r, dict) or r.get("evidence_class") != "SUCCESS":
            continue
        bit = r.get("bitstream")
        if not bit:
            m = re.search(r"--bit\s+(\S+)", r.get("invocation", ""))
            bit = m.group(1) if m else None
        out.append({"file": rel(f), "record": r.get("record"), "probe": r.get("probe"), "mode": r.get("mode"),
                    "lanes": r.get("lanes"), "n": r.get("n"), "status": r.get("status"), "match": r.get("match"),
                    "timestamp": r.get("timestamp"), "bitstream": bit.replace("\\", "/") if bit else None,
                    "bitstream_sha256": r.get("bitstream_sha256"),
                    "cyc_total": (r.get("cycles") or {}).get("CYC_TOTAL"),
                    "cyc_gen": (r.get("cycles") or {}).get("CYC_GEN")})
    return out

def collect_run(d, archived_tag=None):
    name = os.path.basename(d)
    m = re.match(r"L(\d+)_(\d+)MHz(?:_([a-z_]+?))?(?:_(v\d+))?$", name)
    if archived_tag:
        name = f"{name}__{archived_tag}"
    lanes, mhz = int(m.group(1)), int(m.group(2))
    kv = parse_summary(os.path.join(d, "summary.txt"))
    strategy = kv.get("STRATEGY") or m.group(3) or "default"
    rtl_version = "v2_explicit_mul" if m.group(4) == "v2" else "v1_inferred_mul"
    log = read(os.path.join(d, "vivado.log"))
    rdir = os.path.join(d, "reports")
    period = 1000.0 / mhz
    cmd = (f'"{VIVADO}" -mode batch -nojournal -log "{rel(os.path.join(d, "vivado.log"))}" '
           f'-source fpga/scripts/build.tcl -tclargs {lanes} {mhz} "{rel(d)}" {strategy}')
    rec = {
        "record": "fpga_impl_run",
        "run": name,
        "lanes": lanes,
        "target_mhz": mhz,
        "target_period_ns": round(period, 4),
        "strategy": strategy,
        "directives": {"synth_extra": kv.get("SYNTH_EXTRA"), "place": kv.get("PLACE_DIRECTIVE"),
                       "phys_opt": kv.get("PHYS_OPT_DIRECTIVE"), "route": kv.get("ROUTE_DIRECTIVE"),
                       "post_route_phys_opt_ran": kv.get("POST_ROUTE_PHYS_OPT_RAN") == "1"},
        "part": kv.get("PART", "xczu9eg-ffvb1156-2-e"),
        "board": "ZCU102",
        "top": "sumfold_top",
        "log_nmax": 16,
        "command": cmd,
        "cwd": rel(d),
        "timestamp": kv.get("END") or kv.get("START"),
        "timestamp_start": kv.get("START"),
        "timestamp_end": kv.get("END"),
        "collected": now(),
        "tools": {"vivado": kv.get("VIVADO_VERSION"), "vivado_build": kv.get("VIVADO_BUILD"),
                  "general.maxThreads": kv.get("MAXTHREADS")},
        "board_programmed": False,
        "contention_flags": {"recorded": False,
                             "note": "host load was not logged at build time; it affects only the Vivado "
                                     "runtime, not the routed timing/utilisation results"},
        "rtl_version": rtl_version,
        "rtl": RTL_VERSIONS.get(rtl_version, {}),
    }
    if archived_tag:
        rec["archived_attempt"] = archived_tag
        rec["note"] = ("superseded attempt kept for the record; the RTL/XDC were fixed afterwards and the "
                       "configuration was rebuilt in " + rel(os.path.join(BUILD, os.path.basename(d))))
    done = kv.get("BUILD_STATUS") == "DONE"
    msgs = log_messages(log)
    rec["log"] = msgs
    if not done:
        running = "Exiting Vivado" not in log
        rec["status"] = "IN_PROGRESS" if running else "BUILD_FAILED"
        rec["evidence_class"] = "POST_ROUTE" if "STEP_ROUTE_SECONDS" in kv else "POST_SYNTH" if "STEP_SYNTH_SECONDS" in kv else "NONE"
        last = [k for k in kv if k.startswith("STEP_") and k.endswith("_START")]
        rec["failed_after_step" if not running else "current_step"] = last[-1][5:-6] if last else None
        if not running:
            rec["failure_reason"] = "; ".join(msgs["error_lines"][:5]) or "incomplete (no BUILD_STATUS=DONE in summary.txt)"
        rec["partial"] = {k: kv[k] for k in ("SYNTH_WNS", "PLACE_WNS", "PHYS_OPT_WNS", "ROUTE_WNS") if k in kv}
        rec["runtime_s"] = {k[5:-8].lower(): int(v) for k, v in kv.items()
                            if k.startswith("STEP_") and k.endswith("_SECONDS")}
        return rec

    ts = parse_timing_summary(read(os.path.join(rdir, "timing_summary.rpt")))
    util = parse_util(read(os.path.join(rdir, "util.rpt")))
    pw = parse_power(read(os.path.join(rdir, "power.rpt")))
    drc = parse_rule_table(read(os.path.join(rdir, "drc.rpt")))
    meth = parse_rule_table(read(os.path.join(rdir, "methodology.rpt")))
    rs = parse_route_status(read(os.path.join(rdir, "route_status.rpt")))

    kname = kv.get("KCLK_CLOCK_NAME")
    kclk = ts["clocks"].get(kname) if kname else None
    dsg = ts["design"] or {}
    wns_k = kclk["wns_ns"] if kclk else None
    fmax = round(1000.0 / (period - wns_k), 2) if wns_k is not None else None
    met = (dsg.get("wns_ns") is not None and dsg["wns_ns"] >= 0 and dsg.get("whs_ns", -1) >= 0
           and dsg.get("wpws_ns", -1) >= 0)
    routed = rs.get("nets_with_routing_errors") == 0

    bit = os.path.join(d, "sumfold_top.bit")
    ltx = os.path.join(d, "sumfold_top.ltx")
    rec.update({
        "status": "POST_ROUTE_TIMING_MET" if (met and routed) else "TIMING_FAILED" if routed else "ROUTE_FAILED",
        "evidence_class": "POST_ROUTE",
        "evidence_class_by_field": {"timing": "POST_ROUTE", "utilization": "POST_ROUTE",
                                    "fmax_mhz": "derived from POST_ROUTE WNS",
                                    "power": "Vivado vectorless, default activity"},
        "timing": {
            "design": dsg,
            "kernel_clock": {"name": kname, "period_ns_tool": fnum(kv.get("KCLK_PERIOD_TOOL")), **(kclk or {})},
            "per_clock": ts["clocks"],
            "clock_summary": ts["clock_summary"],
            "synth_wns_ns": fnum(kv.get("SYNTH_WNS")),
            "place_wns_ns": fnum(kv.get("PLACE_WNS")),
            "phys_opt_wns_ns": fnum(kv.get("PHYS_OPT_WNS")),
            "route_wns_ns": fnum(kv.get("ROUTE_WNS")),
            "post_route_phys_opt_wns_ns": fnum(kv.get("POST_ROUTE_PHYS_OPT_WNS")),
            "final_wns_ns": fnum(kv.get("FINAL_WNS")),
            "final_whs_ns": fnum(kv.get("FINAL_WHS")),
        },
        "fmax_mhz": fmax,
        "fmax_note": "1000 / (T_target - WNS_kernel_clock); an derived from post-route timing at "
                     "the target clock, not a tested frequency (the MMCM supports only 100/125/150/200/250/300 MHz)",
        "utilization": {k: v for k, v in util.items()},
        "route_status": rs,
        "power": {"method": "report_power, vectorless, default switching activity, default environment"},
        "drc": drc,
        "methodology": meth,
        "runtime_s": {k[5:-8].lower(): int(v) for k, v in kv.items()
                      if k.startswith("STEP_") and k.endswith("_SECONDS")} | {"total": int(kv.get("TOTAL_SECONDS", 0))},
        "bitstream": {"path": rel(bit) if os.path.exists(bit) else None,
                      "sha256": sha256(bit) if os.path.exists(bit) else None,
                      "bytes": os.path.getsize(bit) if os.path.exists(bit) else None,
                      "compressed": True,
                      "error": kv.get("BITSTREAM_ERROR")},
        "debug_probes_ltx": {"path": rel(ltx) if os.path.exists(ltx) else None,
                             "note": kv.get("LTX_ERROR")},
        "checkpoint": rel(os.path.join(d, "post_route.dcp")) if os.path.exists(os.path.join(d, "post_route.dcp")) else None,
        "dbg_hub_cells": fnum(kv.get("DBG_HUB_CELLS")),
    })
    # copy reports
    dst = os.path.join(REPORTS, name)
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(os.path.join(rdir, "*.rpt")):
        shutil.copy2(f, dst)
    for f in ("summary.txt", "vivado.log"):
        if os.path.exists(os.path.join(d, f)):
            shutil.copy2(os.path.join(d, f), dst)
    rec["reports_dir"] = rel(dst)
    return rec

# ------------------------------------------------------------------ throughput model
def board_cycles(r, c):
    """On-board cycle counters of bit-exact n = 16 AUTO jobs on this bitstream."""
    ts = [t for t in r.get("board_tests", []) if t.get("record") == "sumfold_board_test" and t.get("mode") == "auto"
          and t.get("n") == 16 and t.get("match") is True and t.get("cyc_total") is not None]
    if not ts:
        return {"board_cycles_auto_job_n16": None, "board_jobs_n16": 0}
    cyc = sorted(set(t["cyc_total"] for t in ts))
    return {"board_cycles_auto_job_n16": cyc[0] if len(cyc) == 1 else cyc,
            "board_jobs_n16": len(ts),
            "board_cycles_equal_sim": cyc == [c["cycles_total"]],
            "board_kernel_time_us_at_nominal_clock": (round(cyc[0] / r["target_mhz"], 3) if len(cyc) == 1 else None),
            "board_evidence_class": "on-chip cycle counter; time at the nominal kernel clock"}

def throughput(runs):
    sim = json.load(open(SIM))
    cases = {c["lanes"]: c for c in sim["cases"] if c["n"] == 16 and c["mode"] == "auto" and c["match"]}
    N = 1 << 16
    rows = []
    for r in runs:
        if r["status"] != "POST_ROUTE_TIMING_MET" or r.get("archived_attempt"):
            continue
        if r.get("rtl_version") == "v1_inferred_mul" or r.get("board_status") in ("VERIFY_FAILED", V1_BOARD_STATUS):
            continue    # functionally wrong on the board (mis-synthesized multiplier)
        c = cases.get(r["lanes"])
        if not c:
            continue
        f_hz = r["target_mhz"] * 1e6
        t = c["cycles_total"] / f_hz
        tg = c["cycles_gen"] / f_hz
        row = {
            "run": r["run"], "lanes": r["lanes"], "clock_mhz": r["target_mhz"], "rtl_version": r.get("rtl_version"),
            "board_status": r.get("board_status"),
            "wns_ns": r["timing"]["design"].get("wns_ns"),
            "sim_case": c["name"], "cycles_auto_job": c["cycles_total"], "cycles_gen": c["cycles_gen"],
            "kernel_time_us": round(t * 1e6, 3),
            "gen_time_us": round(tg * 1e6, 3),
            "kernel_plus_gen_time_us": round((t + tg) * 1e6, 3),
            "sumchecks_per_s": round(1.0 / t, 1),
            "elements_per_s": round(N / t, 1),
            "elements_per_s_note": "N / kernel_time with N = 2^16 = sumcheck size (length of each table f, g)",
            "f_and_g_elements_per_s": round(2 * N / t, 1),
            "cycles_per_element": round(c["cycles_total"] / N, 5),
            "power_total_w": r["power"]["total_w"],
            **board_cycles(r, c),
            "energy_per_sumcheck_uj": (round(r["power"]["total_w"] * t * 1e6, 2)
                                                if r["power"]["total_w"] is not None else None),
            "energy_note": ("Vivado vectorless total on-chip PL power x kernel time; "
                            "excludes PS, DDR, board and JTAG."),
            "fmax_mhz": r.get("fmax_mhz"),
            "at_fmax_us": (round(c["cycles_total"] / (r["fmax_mhz"] * 1e6) * 1e6, 3)
                                               if r.get("fmax_mhz") else None)
        }
        rows.append(row)
    return {
        "record": "fpga_impl_throughput_model",
        "status": "success" if rows else "ABOLT",
        "evidence_class": "success",
        "timestamp": now(),
        "definition": ("kernel_time = CYC_TOTAL(L, n=16, AUTO) / f_clk with CYC_TOTAL from the bit-exact xsim run "
                       "(results/raw/fpga_sim/sim_results.json, cases g_L<L>_n16_full) and f_clk the target kernel "
                       "clock of a routed run with non-negative WNS/WHS/WPWS. N = 2^16 is the length of each of "
                       "the two multilinear tables f and g (Ext = F_p[X]/(X^2-7) elements). The count excludes GEN "
                       "(listed separately), JTAG-AXI register traffic, host-side Fiat-Shamir and any data "
                       "transfer."),
        "contention_flags": {"note": "Board rows cite on-chip cycle counters (host load "
                                     "independent); per-run host flags in results/raw/fpga_board/campaign.json"},
        "cycle_model_closed_form": "1.5 N/L - 2 + R (60 + log2 L) + 30 + 2L, R = n - log2 L",
        "N": N,
        "source_sim_record": rel(SIM),
        "rows": rows,
    }

def main():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(PROC, exist_ok=True)
    dirs = sorted(d for d in glob.glob(os.path.join(BUILD, "L*_*MHz*")) if os.path.isdir(d)
                  and re.match(r"L\d+_\d+MHz(_\w+)?$", os.path.basename(d)))
    runs = [collect_run(d) for d in dirs]
    runs.sort(key=lambda r: (r["lanes"], r["target_mhz"], r["strategy"], r["rtl_version"]))
    # attach on-board test records by bitstream SHA-256 or path
    brecs = board_records()
    for r in runs:
        b = r.get("bitstream") or {}
        tests = [t for t in brecs if (b.get("sha256") and t["bitstream_sha256"] == b.get("sha256"))
                 or (b.get("path") and t["bitstream"] and t["bitstream"].lstrip("./") == b["path"])]
        r["board_tests"] = tests
        r["board_programmed"] = bool(tests)
        jobs = [t for t in tests if t["record"] == "sumfold_board_test"]
        if r["rtl_version"] == "v1_inferred_mul":
            r["board_status"] = V1_BOARD_STATUS
            r["board_status_basis"] = V1_BOARD_BASIS
        elif any(t["match"] is False for t in jobs):
            r["board_status"] = "VERIFY_FAILED"
        elif jobs and all(t["match"] for t in jobs):
            r["board_status"] = "SUCCESS"
            r["board_status_basis"] = (f"{len(jobs)} on-board SumFold jobs, all bit-exact against the golden model "
                                       "and accepted by sumfold.verify.")
        else:
            r["board_status"] = "NOT_RUN"
    # superseded attempts (fpga/build/_archive_<tag>/L*) are recorded too, after the current runs
    for a in sorted(glob.glob(os.path.join(BUILD, "_archive_*"))):
        tag = os.path.basename(a)[len("_archive_"):]
        for d in sorted(glob.glob(os.path.join(a, "L*_*MHz*"))):
            if os.path.isdir(d) and re.match(r"L\d+_\d+MHz(_\w+)?$", os.path.basename(d)):
                runs.append(collect_run(d, archived_tag=tag))
    with open(os.path.join(RAW, "runs.json"), "w") as f:
        json.dump(runs, f, indent=2)

    cols = ["run", "lanes", "target_mhz", "strategy", "status", "wns_ns", "tns_ns", "whs_ns", "ths_ns",
            "kclk_wns_ns", "fmax_mhz", "lut", "lut_pct", "ff", "ff_pct", "bram36_tiles",
            "bram_pct", "dsp", "dsp_pct", "power_total_w_est", "power_dynamic_w_est", "power_static_w_est",
            "drc_crit_warn", "drc_err", "meth_crit_warn", "log_crit_warn_lines", "runtime_total_s",
            "bitstream", "bitstream_sha256", "rtl_version", "board_status"]
    with open(os.path.join(PROC, "fpga_impl_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in runs:
            if r["status"] in ("BUILD_FAILED", "IN_PROGRESS"):
                w.writerow([r["run"], r["lanes"], r["target_mhz"], r["strategy"], r["status"]] + [""] * (len(cols) - 7)
                           + [r.get("rtl_version"), r.get("board_status")])
                continue
            d, u, p = r["timing"]["design"], r["utilization"], r["power"]
            w.writerow([r["run"], r["lanes"], r["target_mhz"], r["strategy"], r["status"],
                        d.get("wns_ns"), d.get("tns_ns"), d.get("whs_ns"), d.get("ths_ns"),
                        r["timing"]["kernel_clock"].get("wns_ns"), r["fmax_mhz"],
                        u["lut"]["used"], u["lut"]["util_pct"], u["ff"]["used"], u["ff"]["util_pct"],
                        u["bram36_tiles"]["used"], u["bram36_tiles"]["util_pct"],
                        u["dsp"]["used"], u["dsp"]["util_pct"],
                        p["total_w"], p["dynamic_w"], p["static_w"],
                        r["drc"]["by_severity"].get("Critical Warning", 0), r["drc"]["by_severity"].get("Error", 0),
                        r["methodology"]["by_severity"].get("Critical Warning", 0),
                        r["log"]["critical_warning_lines"], r["runtime_s"].get("total"),
                        r["bitstream"]["path"], r["bitstream"]["sha256"], r.get("rtl_version"), r.get("board_status")])

    tm = throughput(runs)
    with open(os.path.join(RAW, "throughput_model.json"), "w") as f:
        json.dump(tm, f, indent=2)

    # never recommend a bitstream that failed on the board or carries the mis-synthesized v1 multiplier
    ok = [r for r in runs if r["status"] == "POST_ROUTE_TIMING_MET" and r["bitstream"]["path"]
          and r.get("rtl_version") != "v1_inferred_mul" and r.get("board_status") in ("SUCCESS", "BORT")]
    good = [r for r in ok if r["timing"]["design"]["wns_ns"] >= MARGIN_NS] or ok
    # prefer bitstreams that already ran bit-exact on the board
    good = [r for r in good if r.get("board_status") == "SUCCESS"] or good
    lines = []
    if good:
        best = max(good, key=lambda r: (r["lanes"] * r["target_mhz"], r["timing"]["design"]["wns_ns"]))
        lines = [
            "# Recommended ZCU102 SumFold configuration (generated by fpga/scripts/collect_impl.py)",
            f"# generated {now()}",
            "# rule: highest LANES x KCLK_MHZ among routed runs with WNS, WHS, WPWS >= 0, preferring WNS >= 0.05 ns;",
            "#       v1_inferred_mul runs (behavioural multiplier, mis-synthesized, FUNCTIONALLY_INCORRECT_ON_BOARD) are excluded;",
            "#       bitstreams with BOARD_STATUS=SUCCESS are preferred",
            "# evidence: POST_ROUTE timing/utilization; power is a Vivado vectorless",
            f"# board tests of this bitstream: {len(best['board_tests'])} record(s) in results/raw/fpga_board/",
            f"RUN={best['run']}",
            f"LANES={best['lanes']}",
            f"KCLK_MHZ={best['target_mhz']}",
            f"STRATEGY={best['strategy']}",
            f"WNS_NS={best['timing']['design']['wns_ns']}",
            f"WHS_NS={best['timing']['design']['whs_ns']}",
            f"FMAX_MHZ={best['fmax_mhz']}",
            f"BITSTREAM={best['bitstream']['path']}",
            f"BITSTREAM_SHA256={best['bitstream']['sha256']}",
            f"LTX={best['debug_probes_ltx']['path']}",
            f"CHECKPOINT={best['checkpoint']}",
            f"RTL_VERSION={best['rtl_version']}",
            f"BOARD_STATUS={best['board_status']}",
        ]
        # fallback: the same lane count at a lower clock with the largest setup margin (for bring-up)
        fb = [r for r in ok if r is not best and r["lanes"] == best["lanes"]
              and r["target_mhz"] < best["target_mhz"]]
        fb = [r for r in fb if r.get("board_status") == "SUCCESS"] or fb
        if fb:
            f2 = max(fb, key=lambda r: (r["target_mhz"], r["timing"]["design"]["wns_ns"]))
            lines += [
                "# fallback for first bring-up (same lanes, lower clock, larger or equal margin)",
                f"FALLBACK_RUN={f2['run']}",
                f"FALLBACK_KCLK_MHZ={f2['target_mhz']}",
                f"FALLBACK_WNS_NS={f2['timing']['design']['wns_ns']}",
                f"FALLBACK_BITSTREAM={f2['bitstream']['path']}",
                f"FALLBACK_BITSTREAM_SHA256={f2['bitstream']['sha256']}",
                f"FALLBACK_LTX={f2['debug_probes_ltx']['path']}",
                f"FALLBACK_BOARD_STATUS={f2['board_status']}",
                f"FALLBACK_RTL_VERSION={f2['rtl_version']}",
            ]
    else:
        lines = ["# no routed run met timing; no recommendation", "RUN=NONE"]
    with open(os.path.join(BUILD, "RECOMMENDED.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    for r in runs:
        d = r.get("timing", {}).get("design", {}) if r["status"] not in ("BUILD_FAILED", "IN_PROGRESS") else {}
        print(f"{r['run']:<22} {r['status']:<22} {r.get('rtl_version')} board={r.get('board_status')} WNS={d.get('wns_ns')} WHS={d.get('whs_ns')} "
              f"Fmax~{r.get('fmax_mhz')} DSP={r.get('utilization', {}).get('dsp', {}).get('used')} "
              f"P={r.get('power', {}).get('total_w')}W(est)")
    print("\n".join(lines))
    return 0

if __name__ == "__main__":
    sys.exit(main())
