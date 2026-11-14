#!/usr/bin/env python3
"""Post-synthesis (netlist) checks of the Goldilocks multiplier gl_mul.

Why: RTL simulation checks the RTL, not what synthesis made of it. On 2026-09-27 the first board run failed because Vivado 2022.2 mis-pipelined the behavioural 64x64 multiplier (two of its sixteen 17x17 partial products one cycle early / late). This script makes that class of bug visible:

  1. synth_unit.tcl : out-of-context synth_design of gl_mul (same part and default options as the
     board build) + write_verilog -mode funcsim, for
       legacy : -verilog_define GL_MUL_LEGACY_INFER (the multiplier of the failing bitstreams)
       fixed  : default RTL (explicit 12-DSP48E2 decomposition)
  2. xsim with unisims: tb_mul_netlist.sv (512 golden vectors at II = 1, and with one random
     operand pair between vectors) and tb_mul_impulse.sv (64 single-bit impulses: every partial
     product must reach y exactly GL_MUL_LAT = 11 cycles later).
  (A gl_mul cell extracted from a routed board checkpoint has optimized-away/renamed ports and cannot
  be driven standalone; whole-design routed netlists are checked with fpga/sim/replay/replay.py
  --dut-netlist instead.)
Writes results/raw/fpga_sim/netlist_gl_mul.json.

Usage: python fpga/sim/netlist/run_netlist_checks.py [--no-synth] [--only legacy,fixed]
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
VIVADO = r"<vivado-2022.2>\bin\vivado.bat"
OUT = os.path.join(ROOT, "results", "raw", "fpga_sim", "netlist_gl_mul.json")
STARTUP = ("rt/data/common.tcl", "rt-undefined", "rtSynthParallelPrep", "Cannot open file:")

def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")

def run(cmd, log):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=HERE, shell=True, capture_output=True, text=True, errors="replace")
    with open(log, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"$ {cmd}\n{p.stdout}{p.stderr}\n# exit={p.returncode}\n")
    return p.returncode, p.stdout + p.stderr, time.time() - t0

def synth(name, extra, commands):
    for attempt in range(4):
        cmd = (f'"{VIVADO}" -mode batch -nojournal -log synth_{name}.log -source synth_unit.tcl '
               f'-tclargs gl_mul {name} {extra}').strip()
        rc, out, el = run(cmd, os.path.join(HERE, f"synth_{name}.stdout"))
        commands.append({"cwd": rel(HERE), "cmd": cmd, "exit": rc, "seconds": round(el, 1)})
        if "NETLIST_DONE" in out:
            return True
        if not any(s in out for s in STARTUP):
            return False
        time.sleep(30)
    return False

def sim(name, commands):
    cmd = f'cmd.exe /c "{os.path.join(HERE, "run_mulnet_dir.cmd")}" {name}'
    rc, out, el = run(cmd, os.path.join(HERE, f"sim_{name}.stdout"))
    commands.append({"cwd": rel(HERE), "cmd": cmd, "exit": rc, "seconds": round(el, 1)})
    w = os.path.join(HERE, f"work_{name}")
    res = {}
    for gap in (0, 1):
        t = open(os.path.join(w, f"xsim_gap{gap}.out"), errors="replace").read() if os.path.exists(os.path.join(w, f"xsim_gap{gap}.out")) else ""
        m = re.search(r"MULNET_RESULT (\w+) gap=(\d+) checked=(\d+) errors=(\d+)", t)
        res[f"vectors_gap{gap}"] = {"result": m.group(1), "checked": int(m.group(3)), "errors": int(m.group(4))} if m else None
    t = open(os.path.join(w, "xsim_imp.out"), errors="replace").read() if os.path.exists(os.path.join(w, "xsim_imp.out")) else ""
    m = re.search(r"MULIMP_RESULT (\w+) pairs=(\d+) bad=(\d+)", t)
    res["impulse"] = {"result": m.group(1), "pairs": int(m.group(2)), "bad": int(m.group(3)),
                      "misaligned": re.findall(r"IMP (p=\d+ q=\d+ t=\d+) .*WRONG", t)} if m else None
    dsp = os.path.join(HERE, name, "dsp.rpt")
    res["dsp48e2"] = len(open(dsp).read().splitlines()) if os.path.exists(dsp) else None
    res["netlist"] = rel(os.path.join(HERE, name, "gl_mul_funcsim.v"))
    res["pass"] = bool(res["impulse"] and res["impulse"]["result"] == "PASS"
                       and all(res[k] and res[k]["result"] == "PASS" for k in ("vectors_gap0", "vectors_gap1")))
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-synth", action="store_true")
    ap.add_argument("--only", default="legacy,fixed")
    a = ap.parse_args()
    only = set(a.only.split(","))
    commands, results = [], {}
    targets = []
    if "legacy" in only:
        targets.append(("gl_mul_legacy", "-verilog_define GL_MUL_LEGACY_INFER", "OOC synth, GL_MUL_LEGACY_INFER (multiplier of the failing v1 bitstreams)"))
    if "fixed" in only:
        targets.append(("gl_mul_fix", "", "OOC synth, default RTL (explicit 12-DSP48E2 multiplier)"))
    for name, extra, what in targets:
        ok = True if a.no_synth else synth(name, extra, commands)
        results[name] = {"what": what, "synth_ok": ok, **(sim(name, commands) if ok else {})}
        print(name, json.dumps({k: results[name].get(k) for k in ("pass", "vectors_gap0", "impulse")}))
    rec = {"record": "sumfold_netlist_gl_mul", "evidence_class": "SIMULATION (post-synthesis functional netlist)",
           "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "tools": {"vivado": "2022.2", "simulator": "xsim 2022.2 + unisims_ver"},
           "part": "xczu9eg-ffvb1156-2-e", "latency_expected": 11,
           "results": results, "commands": commands}
    prev = json.load(open(OUT)) if os.path.exists(OUT) else None
    if prev and only != {"legacy", "fixed"}:
        prev["results"].update(results)
        prev["commands"] += commands
        prev["timestamp"] = rec["timestamp"]
        rec = prev
    # expected: legacy FAILS (reproduces the bug), fixed PASSES
    ok = results.get("gl_mul_fix", {}).get("pass", True) and not results.get("gl_mul_legacy", {}).get("pass", False)
    rec["expected_outcome_met"] = ok
    with open(OUT, "w", newline="\n") as fh:
        json.dump(rec, fh, indent=1)
    print("wrote", OUT, "expected outcome met:", ok)
    return 0 if ok else 2

if __name__ == "__main__":
    sys.exit(main())
