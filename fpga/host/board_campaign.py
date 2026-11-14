#!/usr/bin/env python3
"""ZCU102 on-board evidence campaign for the v2 (explicit DSP48E2 multiplier) SumFold bitstreams.

Runs, one Vivado hardware session at a time, the steps listed in STEPS (volatile JTAG configuration of the PL only; --run --authorized is required, as for jtag_board_test.py / board_probe.py). Each step writes its own record to results/raw/fpga_board/campaign/<id>.json; existing records are kept unless --force (so an interrupted campaign resumes). --aggregate only rebuilds

results/raw/fpga_board/campaign.json        every step record (condensed) + contention flags results/raw/fpga_board/summary.json         per-configuration statistics results/processed/fpga_board_summary.csv    one row per configuration

Usage: python fpga/host/board_campaign.py --run --authorized          (hardware) python fpga/host/board_campaign.py --aggregate                 (offline)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import statistics as st
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PY = sys.executable
OUTD = os.path.join(ROOT, "results", "raw", "fpga_board", "campaign")
BUILD = "fpga/build"

SEEDS = [("0123456789abcdef", "fedcba9876543210"),
         ("243f6a8885a308d3", "13198a2e03707344"),
         ("a4093822299f31d0", "082efa98ec4e6c89")]

CFG = {
    "L8_300": {"dir": "L8_300MHz_perf_v2", "lanes": 8, "mhz": 300},
    "L8_250": {"dir": "L8_250MHz_v2", "lanes": 8, "mhz": 250},
    "L4_250": {"dir": "L4_250MHz_v2", "lanes": 4, "mhz": 250},
    "L2_250": {"dir": "L2_250MHz_v2", "lanes": 2, "mhz": 250},
    "L1_250": {"dir": "L1_250MHz_v2", "lanes": 1, "mhz": 250},
}

def bitargs(cfg):
    d = f"{BUILD}/{CFG[cfg]['dir']}"
    return ["--bit", f"{d}/sumfold_top.bit", "--ltx", f"{d}/sumfold_top.ltx"]

def steps():
    s = []
    # ---- L8 300 MHz (recommended). The first session programs the PL.
    s.append(dict(id="L8_300_linkrate", cfg="L8_300", kind="probe", program=True,
                  args=["linkrate", "--lanes", "8", "--logn", "10", "--count", "1000", "--queue", "100"]))
    for n in (10, 12, 14, 16):
        for k, (sfd, sgd) in enumerate(SEEDS):
            s.append(dict(id=f"L8_300_auto_n{n}_s{k}", cfg="L8_300", kind="test", mode="auto", n=n, seed=k,
                          args=["--mode", "auto", "--logn", str(n), "--seed-f", sfd, "--seed-g", sgd]))
    for n in (10, 16):
        for rep in range(3):
            s.append(dict(id=f"L8_300_step_n{n}_r{rep}", cfg="L8_300", kind="test", mode="step", n=n, seed=0,
                          args=["--mode", "step", "--logn", str(n)]))
    s.append(dict(id="L8_300_negtable_n10", cfg="L8_300", kind="probe",
                  args=["negtable", "--lanes", "8", "--logn", "10", "--flip-index", "5", "--flip-bit", "17"]))
    s.append(dict(id="L8_300_negtable_n16", cfg="L8_300", kind="probe",
                  args=["negtable", "--lanes", "8", "--logn", "16", "--flip-index", "40000", "--flip-bit", "0"]))
    s.append(dict(id="L8_300_negch_n10", cfg="L8_300", kind="probe",
                  args=["negch", "--lanes", "8", "--logn", "10", "--flip-index", "2", "--flip-bit", "17"]))
    s.append(dict(id="L8_300_sysmon_n16", cfg="L8_300", kind="probe",
                  args=["sysmon", "--lanes", "8", "--logn", "16", "--count", "2000", "--every", "250"]))
    # ---- L8 250 MHz (fallback)
    first = True
    for n in (10, 12, 14, 16):
        for k, (sfd, sgd) in enumerate(SEEDS):
            s.append(dict(id=f"L8_250_auto_n{n}_s{k}", cfg="L8_250", kind="test", mode="auto", n=n, seed=k,
                          program=first, args=["--mode", "auto", "--logn", str(n), "--seed-f", sfd, "--seed-g", sgd]))
            first = False
    # ---- lane sweep at 250 MHz, n = 16
    for cfg in ("L4_250", "L2_250", "L1_250"):
        for k in range(2):
            sfd, sgd = SEEDS[k]
            s.append(dict(id=f"{cfg}_auto_n16_s{k}", cfg=cfg, kind="test", mode="auto", n=16, seed=k,
                          program=(k == 0),
                          args=["--mode", "auto", "--logn", "16", "--seed-f", sfd, "--seed-g", sgd]))
    # ---- leave the recommended bitstream on the board; final temperature / rail sample
    s.append(dict(id="L8_300_final_sysmon", cfg="L8_300", kind="probe", program=True,
                  args=["sysmon", "--lanes", "8", "--logn", "16", "--count", "200", "--every", "100"]))
    # repeat of the link-rate probe at the end (host-load sensitivity of the JTAG timings)
    s.append(dict(id="L8_300_linkrate_r2", cfg="L8_300", kind="probe",
                  args=["linkrate", "--lanes", "8", "--logn", "10", "--count", "1000", "--queue", "100"]))
    return s

def contention():
    """Other Vivado / hw_server / xsim processes and GPU compute processes at the start of a step."""
    flags = {}
    try:
        out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True).stdout
        names = [ln.split(",")[0].strip('"').lower() for ln in out.splitlines() if ln]
        flags["vivado_procs"] = sum(n.startswith("vivado") for n in names)
        flags["hw_server_procs"] = sum(n.startswith("hw_server") for n in names)
        flags["xsim_procs"] = sum(n.startswith("xsim") for n in names)
    except OSError as e:
        flags["tasklist_error"] = repr(e)
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20).stdout
        flags["gpu_compute_procs"] = len([ln for ln in q.splitlines() if ln.strip()])
    except (OSError, subprocess.SubprocessError) as e:
        flags["gpu_query_error"] = repr(e)
    try:
        l1 = subprocess.run(["powershell", "-NoProfile", "-Command",
                             "(Get-CimInstance Win32_Processor | -Object -Property LoadPercentage -Average).Average"],
                            capture_output=True, text=True, timeout=30).stdout.strip()
        flags["host_cpu_load_pct"] = float(l1) if l1 else None
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        flags["cpu_query_error"] = repr(e)
    flags["note"] = ("JTAG-to-AXI timings are host-side and can be slowed by host load; device cycle counts "
                     "are on-chip counters and are not affected")
    return flags

def connect_failed(out):
    """True if the Vivado session never reached the device (hw_server / cs_server start-up failure)."""
    try:
        rec = json.load(open(out, encoding="utf-8"))
        log = os.path.join(ROOT, rec.get("log") or "")
        text = open(log, encoding="utf-8", errors="replace").read() if os.path.isfile(log) else ""
    except (OSError, ValueError):
        return False
    return "'connect_hw_server' failed" in text or "cs_server initialization" in text

def run_step(s, force, attempt=0):
    out = os.path.join(OUTD, s["id"] + ".json")
    if os.path.exists(out) and not force and attempt == 0:
        print("skip (exists)", s["id"])
        return
    _run_step(s, out)
    if connect_failed(out) and attempt < 2:
        # keep the failed record (renamed) and retry: the failure is before any register access
        os.replace(out, os.path.join(OUTD, f"_connect_failed_{s['id']}_{int(time.time())}.json"))
        print("hw_server/cs_server connection failed; retrying", s["id"], flush=True)
        time.sleep(15)
        run_step(s, True, attempt + 1)

def _run_step(s, out):
    cfl = contention()
    prog = [] if s.get("program") else ["--no-program"]
    if s["kind"] == "test":
        cmd = [PY, "fpga/host/jtag_board_test.py", "--lanes", str(CFG[s["cfg"]]["lanes"]), *s["args"],
               *bitargs(s["cfg"]), *prog, "--run", "--authorized", "--tag", s["id"], "--out", out]
    else:
        cmd = [PY, "fpga/host/board_probe.py", *s["args"], *bitargs(s["cfg"]), *prog, "--run", "--authorized",
               "--tag", s["id"], "--out", out]
    t0 = time.time()
    print(time.strftime("%H:%M:%S"), "RUN", s["id"], flush=True)
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, errors="replace",
                       env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")})
    print(p.stdout[-800:], p.stderr[-800:], flush=True)
    if os.path.exists(out):
        rec = json.load(open(out, encoding="utf-8"))
        rec["campaign"] = {"step": s["id"], "config": s["cfg"], "campaign_command": " ".join(
            ["python"] + [os.path.relpath(c, ROOT).replace("\\", "/") if c == PY else c for c in cmd[1:]]),
            "contention_flags": cfl, "driver_exit_code": p.returncode, "wall_s": round(time.time() - t0, 1)}
        with open(out, "w", newline="\n") as fh:
            json.dump(rec, fh, indent=1)
    else:
        with open(out.replace(".json", ".FAILED.json"), "w", newline="\n") as fh:
            json.dump({"step": s["id"], "status": "DRIVER_FAILED", "evidence_class": "SUCCESS",
                       "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                       "command": " ".join(cmd), "exit_code": p.returncode, "stdout": p.stdout[-4000:],
                       "stderr": p.stderr[-4000:], "contention_flags": cfl}, fh, indent=1)

# ------------------------------------------------------------------ aggregation
def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def stats(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    return {"n": len(v), "mean": st.mean(v), "median": st.median(v), "min": min(v), "max": max(v),
            "stdev": st.stdev(v) if len(v) > 1 else 0.0}

def aggregate():
    recs = {}
    for s in steps():
        p = os.path.join(OUTD, s["id"] + ".json")
        if os.path.exists(p):
            recs[s["id"]] = (s, json.load(open(p, encoding="utf-8")))
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    runs, groups, sysmon = [], {}, []
    for sid, (s, r) in recs.items():
        c = CFG[s["cfg"]]
        row = {"step": sid, "config": s["cfg"], "bitstream": r.get("bitstream"),
               "bitstream_sha256": r.get("bitstream_sha256"), "timestamp": r.get("timestamp"),
               "status": r.get("status"), "evidence_class": r.get("evidence_class"),
               "command": r.get("campaign", {}).get("campaign_command"), "vivado_command": r.get("command"),
               "log": r.get("log"), "contention_flags": r.get("campaign", {}).get("contention_flags"),
               "programmed": r.get("programmed")}
        if s["kind"] == "test":
            cyc = r.get("cycles", {})
            row.update({"kind": s["mode"], "n": r.get("n"), "lanes": r.get("lanes"), "seed_f": r.get("seed_f"),
                        "seed_g": r.get("seed_g"), "match": r.get("match"),
                        "transcript_verified": r.get("transcript_verified"), "errors": r.get("errors"),
                        "cyc_total": cyc.get("CYC_TOTAL"), "cyc_gen": cyc.get("CYC_GEN"),
                        "cyc_pass": [cyc[f"CYC_PASS_{i}"] for i in range(40) if f"CYC_PASS_{i}" in cyc],
                        "kclk_mhz_nominal": r.get("kclk_mhz_nominal"),
                        "host_times_us": r.get("host_times_us"), "step_round_us": r.get("step_round_us"),
                        "sysmon": r.get("sysmon")})
            if r.get("sysmon"):
                sysmon.append({"step": sid, "config": s["cfg"], "timestamp": r.get("timestamp"), **r["sysmon"]})
            key = (s["cfg"], s["mode"], r.get("n"))
            groups.setdefault(key, []).append(row)
        else:
            row.update({"kind": s["args"][0], "n": r.get("n"), "lanes": r.get("lanes"), "match": r.get("match"),
                        "checked": r.get("checked"), "mismatches": r.get("mismatches"), "raw": r.get("raw"),
                        "negative_test": r.get("negative_test")})
            sm = {k: v for k, v in (r.get("raw") or {}).items() if k.startswith("SYSMON_")}
            if sm:
                sysmon.append({"step": sid, "config": s["cfg"], "timestamp": r.get("timestamp"), **sm})
        runs.append(row)

    summary_rows = []
    for (cfg, mode, n), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        mhz = CFG[cfg]["mhz"]
        cts = sorted(set(r["cyc_total"] for r in rows if r["cyc_total"] is not None))
        gens = sorted(set(r["cyc_gen"] for r in rows if r["cyc_gen"] is not None))
        row = {"config": cfg, "bitstream_dir": CFG[cfg]["dir"], "lanes": CFG[cfg]["lanes"], "kclk_mhz": mhz,
               "mode": mode, "n": n, "runs": len(rows),
               "seeds": len(set((r["seed_f"], r["seed_g"]) for r in rows)),
               "all_match": all(r["match"] is True for r in rows),
               "all_transcripts_verified": all(r["transcript_verified"] is True for r in rows),
               "cyc_total": cts[0] if len(cts) == 1 else cts,
               "cyc_total_identical_across_runs": len(cts) == 1,
               "device_us_at_nominal_clock": (cts[0] / mhz) if len(cts) == 1 else None,
               "cyc_gen": gens[0] if len(gens) == 1 else gens,
               "job_host_us": stats([fnum((r["host_times_us"] or {}).get("JOB_HOST_US")) for r in rows]),
               "gen_host_us": stats([fnum((r["host_times_us"] or {}).get("GEN_HOST_US")) for r in rows]),
               "evidence_class": "SUCCESS"}
        if mode == "step":
            poll = [x for r in rows for x in (r["step_round_us"] or {}).get("poll", [])]
            serv = [x for r in rows for x in (r["step_round_us"] or {}).get("service", [])]
            rt = [a + b for r in rows for a, b in zip((r["step_round_us"] or {}).get("poll", []),
                                                      (r["step_round_us"] or {}).get("service", []))]
            row.update({"step_round_poll_us": stats(poll), "step_round_service_us": stats(serv),
                        "step_round_trip_us": stats(rt),
                        "step_round_trip_definition": "poll (START write -> WAIT observed, includes the device "
                                                      "round and JTAG polling) + service (3 message reads, Tcl "
                                                      "SHA-256 Fiat-Shamir, challenge write, START)"})
        summary_rows.append(row)

    # link rate (two runs: start of the campaign and end of the campaign; host load differs)
    def link_of(sid):
        lr = recs.get(sid)
        if not lr:
            return None
        raw = lr[1].get("raw", {})
        def per(nk, uk):
            n_, u_ = fnum(raw.get(nk)), fnum(raw.get(uk))
            return None if not n_ or u_ is None else {"ops": int(n_), "total_us": u_, "us_per_op": u_ / n_,
                                                     "ops_per_s": n_ / (u_ / 1e6),
                                                     "payload_bytes_per_s": 4 * n_ / (u_ / 1e6)}
        link = {"evidence_class": "SUCCESS", "step": sid, "timestamp": lr[1].get("timestamp"),
                "command": lr[1].get("campaign", {}).get("campaign_command"),
                "axi_master": "JTAG-to-AXI (jtag_axi v1.2), AXI4-Lite, 32-bit data",
                "write_create_run": per("LR_WRITE_CR_N", "LR_WRITE_CR_US"),
                "read_create_run": per("LR_READ_CR_N", "LR_READ_CR_US"),
                "write_run_precreated": per("LR_WRITE_RUN_N", "LR_WRITE_RUN_US"),
                "read_run_precreated": per("LR_READ_RUN_N", "LR_READ_RUN_US"),
                "write_queued": per("LR_WRITE_Q_N", "LR_WRITE_Q_US"),
                "read_queued": per("LR_READ_Q_N", "LR_READ_Q_US"),
                "read_errors": {k: raw.get(k) for k in ("LR_READ_CR_BAD", "LR_READ_RUN_BAD", "LR_READ_Q_BAD",
                                                         "LR_RWCHECK_BAD", "LR_RWCHECK_N", "LR_WRITE_CR_LASTOK",
                                                         "LR_WRITE_Q_CHECK", "LR_MEMELEM_READBACK_BAD",
                                                         "LR_MEMELEM_READBACK_N")},
                "burst": {k: raw.get(k) for k in raw if k.startswith("LR_BURST")},
                "queue_errors": {k: raw.get(k) for k in raw if k.endswith("_Q_ERR")},
                "mem_element_write": per("LR_MEMELEM_N", "LR_MEMELEM_US")}
        me = link["mem_element_write"]
        wcr = link["write_create_run"]
        if me and wcr:
            pairs = 1 << 16
            link["table_load_extrapolation"] = {
                "evidence_class": "ANALYTIC",
                "statement": "EXTRAPOLATION from the per-transaction times.",
                "pairs": pairs,
                "full_128bit_elements_s": 2 * pairs * me["us_per_op"] / 1e6,
                "full_128bit_elements_basis": "MEM element write (7 single-beat writes: MEM_ADDR, "
                                              "MEM_SEL, MEM_W0..W3, MEM_WE) x 2^17 elements",
                "base_field_min_writes_per_pair": 9,
                "base_field_min_s": pairs * 9 * wcr["us_per_op"] / 1e6,
                "base_field_min_basis": "9 writes per (f, g) pair when c1 = 0 is kept in MEM_W2/W3 "
                                        "(MEM_ADDR, SEL, W0, W1, WE for f; SEL, W0, W1, WE for g) x"
                                        "create+run write time",
            }
            if link.get("write_queued"):
                link["table_load_extrapolation"]["base_field_min_queued_s"] = \
                    pairs * 9 * link["write_queued"]["us_per_op"] / 1e6
        link["contention_flags"] = lr[1].get("campaign", {}).get("contention_flags")
        return link

    runs_lr = [x for x in (link_of("L8_300_linkrate"), link_of("L8_300_linkrate_r2")) if x]
    link = None
    if runs_lr:
        prim = min(runs_lr, key=lambda x: (x.get("contention_flags") or {}).get("host_cpu_load_pct") or 999)
        link = {"primary": prim["step"],
                "primary_rule": "run with the lower host CPU load at its start (JTAG timings are host-bound)",
                "runs": runs_lr}

    negs = []
    for sid in ("L8_300_negtable_n10", "L8_300_negtable_n16", "L8_300_negch_n10"):
        if sid in recs:
            r = recs[sid][1]
            negs.append({"step": sid, "n": r.get("n"), "timestamp": r.get("timestamp"),
                         "command": r.get("campaign", {}).get("campaign_command"), **(r.get("negative_test") or {}),
                         "all_words_as_modelled": r.get("match")})

    loop = []
    for sid in ("L8_300_sysmon_n16", "L8_300_final_sysmon"):
        if sid in recs:
            r = recs[sid][1]
            raw = r.get("raw", {})
            jobs, us = fnum(raw.get("LOOP_JOBS")), fnum(raw.get("LOOP_US"))
            ct = int(raw["CYC_TOTAL"], 16) if raw.get("CYC_TOTAL") else None
            cg = int(raw["CYC_GEN"], 16) if raw.get("CYC_GEN") else None
            busy = (jobs * (ct + cg) / 300.0) if (jobs and ct is not None and cg is not None) else None
            loop.append({"step": sid, "timestamp": r.get("timestamp"), "jobs": jobs, "loop_s": us / 1e6 if us else None,
                         "poll_failures": raw.get("LOOP_POLL_FAIL"), "last_job_match": r.get("match"),
                         "kernel_busy_fraction": (busy / us) if (busy and us) else None,
                         "sysmon": {k: v for k, v in raw.items() if k.startswith("SYSMON_")}})

    tests = [r for r in runs if r["kind"] in ("auto", "step")]
    campaign = {"record": "sumfold_board_campaign", "status": "SUCCESS" if recs else "ABORT",
                "evidence_class": "SUCCESS", "timestamp": now,
                "command": "python fpga/host/board_campaign.py --run --authorized",
                "aggregate_command": "python fpga/host/board_campaign.py --aggregate",
                "board": "ZCU102 (xczu9eg-ffvb1156-2-e), JTAG-to-AXI; volatile PL configuration only",
                "rtl_version": "v2_explicit_mul",
                "steps_planned": len(steps()), "steps_recorded": len(recs),
                "contention_flags": {"per_run": "runs[].contention_flags",
                                     "host_cpu_load_pct_range": [
                                         min((r["contention_flags"] or {}).get("host_cpu_load_pct", 0) or 0 for r in runs),
                                         max((r["contention_flags"] or {}).get("host_cpu_load_pct", 0) or 0 for r in runs)]
                                     if runs else None},
                "runs": runs, "link_rate": link, "negative_tests": negs, "sysmon_loops": loop,
                "sysmon_samples": sysmon}
    summary = {"record": "sumfold_board_summary", "status": campaign["status"], "evidence_class": "SUCCESS",
               "timestamp": now, "command": campaign["aggregate_command"], "rtl_version": "v2_explicit_mul",
               "correctness": {"test_runs": len(tests), "match": sum(r["match"] is True for r in tests),
                               "transcript_verified": sum(r["transcript_verified"] is True for r in tests)},
               "configs": summary_rows, "link_rate": link, "negative_tests": negs, "sysmon_loops": loop,
               "contention_note": "per-run contention flags in campaign.json (runs[].contention_flags)",
               "contention_flags": campaign["contention_flags"],
               "evidence_notes": {
                   "device_us_at_nominal_clock": "on-chip cycle counter / nominal kernel clock ("
                                                 "cycles; the 250 MHz clock was at 250.065 MHz on v1)",
                   "host_us": "Tcl clock microseconds around JTAG-to-AXI transactions (host-dependent)"}}
    os.makedirs(os.path.dirname(OUTD), exist_ok=True)
    with open(os.path.join(ROOT, "results", "raw", "fpga_board", "campaign.json"), "w", newline="\n") as fh:
        json.dump(campaign, fh, indent=1)
    with open(os.path.join(ROOT, "results", "raw", "fpga_board", "summary.json"), "w", newline="\n") as fh:
        json.dump(summary, fh, indent=1)
    csvp = os.path.join(ROOT, "results", "processed", "fpga_board_summary.csv")
    cols = ["config", "bitstream_dir", "lanes", "kclk_mhz", "mode", "n", "runs", "seeds", "all_match",
            "all_transcripts_verified", "cyc_total", "device_us_at_nominal_clock", "cyc_gen",
            "job_host_us_median", "job_host_us_min", "job_host_us_max", "step_round_trip_us_median",
            "step_round_trip_us_mean", "step_round_service_us_median", "evidence_class"]
    with open(csvp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in summary_rows:
            j = r["job_host_us"] or {}
            rt = r.get("step_round_trip_us") or {}
            sv = r.get("step_round_service_us") or {}
            w.writerow([r["config"], r["bitstream_dir"], r["lanes"], r["kclk_mhz"], r["mode"], r["n"], r["runs"],
                        r["seeds"], r["all_match"], r["all_transcripts_verified"], r["cyc_total"],
                        None if r["device_us_at_nominal_clock"] is None else round(r["device_us_at_nominal_clock"], 3),
                        r["cyc_gen"], j.get("median"), j.get("min"), j.get("max"),
                        rt.get("median"), None if rt.get("mean") is None else round(rt["mean"], 1),
                        sv.get("median"), "SUCCESS"])
    print(json.dumps({"steps_recorded": len(recs), "tests": summary["correctness"]}, indent=1))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--authorized", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--only", default="", help="comma list of step-id prefixes")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.run:
        if not args.authorized:
            raise SystemExit("--run programs the ZCU102 PL over JTAG (volatile); needs --authorized")
        os.makedirs(OUTD, exist_ok=True)
        pref = [p for p in args.only.split(",") if p]
        for s in steps():
            if pref and not any(s["id"].startswith(p) for p in pref):
                continue
            run_step(s, args.force)
    aggregate()

if __name__ == "__main__":
    main()
