#!/usr/bin/env python3
"""Replay the exact register transactions of a generated board script in xsim.

1. TRACE : the generated hardware Tcl (fpga/host/generated/board_test_<mode>_L<L>_n<n>.tcl) is run in
           Vivado batch mode with the AXI procs replaced by a recorder (no hardware manager is opened):
           every axi_wr / axi_rd / poll_status call becomes one line W/R/P of <work>/trace.txt, in the
           order and with the values of the board run (AUTO mode only: STEP challenges depend on the
           device's messages, so a static trace does not exist for STEP).
2. SIM   : fpga/tb/tb_axi_replay.sv drives sumfold_axil + sumfold_core with that trace (xsim).
           --mul-netlist <file> substitutes a synthesized gl_mul netlist (write_verilog -mode funcsim)
           for the RTL gl_mul, to check whether the post-synthesis multiplier reproduces board results.
3. PLAY  : the same Tcl script is run again with axi_rd / poll_status answering, in order, with the
           values read in the simulation; it prints exactly the SFB_* lines a board run would print, and
           fpga/host/jtag_board_test.py::check() compares them with sumfold.golden.

Usage: python fpga/sim/replay/replay.py --tcl fpga/host/generated/board_test_auto_L8_n10.tcl --lanes 8 --kclk 250 --tag L8_n10_rtl python fpga/sim/replay/replay.py ... --mul-netlist fpga/sim/netlist/gl_mul/gl_mul_funcsim.v --tag L8_n10_mulnet python fpga/sim/replay/replay.py --trace-only ...        (only step 1) --probe-trace <file> : use a trace written by fpga/host/board_probe.py --trace instead of steps 1 and 3
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "fpga", "host"))

from edgecert import sumfold as sf          # noqa: E402
import jtag_board_test as jbt               # noqa: E402

VBIN = r"<vivado-2022.2>\bin"
VIVADO = os.path.join(VBIN, "vivado.bat")
RTL = ["gl_arith.v", "ext_arith.v", "sdp_bram.v", "adder_tree.v", "sumfold_lane.v", "sumfold_core.v", "sumfold_axil.v"]
MARK = "# ---------------------------------------------------------------- hardware bring-up"

INJECT = r"""
# ---- injected by fpga/sim/replay/replay.py (mode @MODE@): AXI access recorder / player
set DRY_RUN 1
set ::RP_MODE {@MODE@}
if {$::RP_MODE eq "trace"} {
    set ::RP_TR [open {@TRACE@} w]
} else {
    set fh [open {@VALUES@} r]
    set ::RP_VALS [split [string trim [read $fh]] "\n"]
    close $fh
    set ::RP_I 0
}
proc rp_next {} {
    set v [lindex $::RP_VALS $::RP_I]
    incr ::RP_I
    if {$v eq ""} { kv ERROR "replay: ran out of recorded read values at read $::RP_I"; return 0 }
    return [expr "0x$v"]
}
proc axi_wr {addr data} {
    if {$::RP_MODE eq "trace"} {
        puts $::RP_TR [format "W %08x %08x" [expr {$addr + 0}] [expr {$data & 0xFFFFFFFF}]]
    }
    emu_wr $addr $data
}
proc axi_rd {addr} {
    if {$::RP_MODE eq "trace"} {
        puts $::RP_TR [format "R %08x 00000000" [expr {$addr + 0}]]
        return [emu_rd $addr]
    }
    return [rp_next]
}
proc poll_status {mask what} {
    global REG
    if {$::RP_MODE eq "trace"} {
        puts $::RP_TR [format "P %08x %08x" [expr {$REG(STATUS) + 0}] [expr {$mask + 0}]]
        return [emu_rd $REG(STATUS)]
    }
    set s [rp_next]
    if {$s & 0x4} { kv ERROR "STATUS.ERROR while waiting for $what (STATUS=[format %08X $s])"; return $s }
    if {!($s & $mask)} { kv ERROR "timeout waiting for $what" ; return -1 }
    return $s
}
"""

TAIL = r"""
if {$::RP_MODE eq "trace"} { close $::RP_TR }
kv REPLAY_READS_USED [expr {$::RP_MODE eq "trace" ? -1 : $::RP_I}]
"""

def fwd(p: str) -> str:
    return p.replace("\\", "/")

def sh(cmd: str, cwd: str, log: str) -> tuple[int, str, float]:
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True, errors="replace")
    out = p.stdout + p.stderr
    with open(log, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"$ {cmd}\n{out}\n# exit={p.returncode}\n")
    return p.returncode, out, time.time() - t0

def make_tcl(src: str, mode: str, trace: str, values: str) -> str:
    txt = open(src, encoding="utf-8").read()
    if MARK not in txt:
        raise SystemExit("marker line not found in the generated Tcl")
    inj = INJECT.replace("@MODE@", mode).replace("@TRACE@", fwd(trace)).replace("@VALUES@", fwd(values))
    txt = txt.replace(MARK, inj + "\n" + MARK, 1)
    # drop the final 'kv END 1' marker and add ours after the tail
    txt = txt.rstrip() + "\n" + TAIL
    return txt

def tcl_param(txt: str, name: str) -> str:
    m = re.search(rf"^set {name}\s+\{{?([^}}\n]*)\}}?", txt, flags=re.M)
    return m.group(1).strip() if m else ""

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tcl", default="")
    ap.add_argument("--probe-trace", default="")
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--kclk", type=int, default=250)
    ap.add_argument("--lognmax", type=int, default=16)
    ap.add_argument("--mul-netlist", default="")
    ap.add_argument("--extra-rtl", default="", help="comma-separated extra Verilog files compiled after the RTL")
    ap.add_argument("--dut-netlist", default="", help="funcsim netlist of sumfold_axil (write_verilog -cell u_sumfold) used instead of the RTL")
    ap.add_argument("--wmode", type=int, default=0)
    ap.add_argument("--wdly", type=int, default=2)
    ap.add_argument("--gap", type=int, default=8)
    ap.add_argument("--poweron0", action="store_true", help="zero challenge/result RAMs and lane memories at t=0")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--trace-only", action="store_true")
    ap.add_argument("--reuse-snapshot", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--snapshot-from", default="", help="copy xsim.dir from this work directory (with --reuse-snapshot)")
    args = ap.parse_args()

    work = os.path.join(HERE, f"work_{args.tag}")
    if args.snapshot_from and not os.path.exists(os.path.join(work, "xsim.dir")):
        shutil.copytree(os.path.join(args.snapshot_from, "xsim.dir"), os.path.join(work, "xsim.dir"))
    os.makedirs(work, exist_ok=True)
    trace = os.path.join(work, "trace.txt")
    values = os.path.join(work, "values.txt")
    rec = {"record": "sumfold_axi_replay", "evidence_class": "SIMULATION", "tag": args.tag,
           "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "invocation": "python " + " ".join([fwd(os.path.relpath(__file__, ROOT))] + sys.argv[1:]),
           "lanes": args.lanes, "kclk_mhz_param": args.kclk, "mul_netlist": args.mul_netlist or None,
           "dut_netlist": fwd(os.path.relpath(os.path.abspath(args.dut_netlist), ROOT)) if args.dut_netlist else None,
           "bus": {"wmode": args.wmode, "wdly": args.wdly, "gap": args.gap}, "poweron0": args.poweron0, "commands": []}

    # ------------------------------------------------------------ 1. trace
    if args.probe_trace:
        shutil.copyfile(args.probe_trace, trace)
        src_txt = ""
    else:
        src_txt = open(args.tcl, encoding="utf-8").read()
        if tcl_param(src_txt, "MODE") != "auto":
            raise SystemExit("only AUTO scripts have a static transaction trace")
        ttcl = os.path.join(work, "trace_mode.tcl")
        with open(ttcl, "w", newline="\n") as fh:
            fh.write(make_tcl(args.tcl, "trace", trace, values))
        cmd = f'"{VIVADO}" -mode batch -nojournal -log "{os.path.join(work, "trace_mode.log")}" -source "{ttcl}"'
        rc, out, el = sh(cmd, work, os.path.join(work, "trace_mode.stdout"))
        rec["commands"].append(cmd)
        if rc != 0 or not os.path.exists(trace):
            print(out[-3000:])
            return 1
        rec["source_tcl"] = fwd(os.path.relpath(args.tcl, ROOT))
    ops = [ln.split() for ln in open(trace) if ln.strip()]
    rec["trace"] = {"file": fwd(os.path.relpath(trace, ROOT)), "ops": len(ops),
                    "writes": sum(o[0] == "W" for o in ops), "reads": sum(o[0] == "R" for o in ops),
                    "polls": sum(o[0] == "P" for o in ops)}
    print("trace:", rec["trace"])
    if args.trace_only:
        return 0

    # ------------------------------------------------------------ 2. simulation
    rtl = " ".join(fwd(os.path.join(ROOT, "fpga", "rtl", f)) for f in RTL)
    inc = fwd(os.path.join(ROOT, "fpga", "rtl"))
    tb = fwd(os.path.join(ROOT, "fpga", "tb", "tb_axi_replay.sv"))
    xv, xe, xs = (f'"{os.path.join(VBIN, t + ".bat")}"' for t in ("xvlog", "xelab", "xsim"))
    snap = "tb_replay"
    if not args.reuse_snapshot:
        cmds = [f"{xv} -i {inc} {rtl}"] if not args.dut_netlist else [f"{xv} {fwd(os.path.abspath(args.dut_netlist))}"]
        extra = [e for e in args.extra_rtl.split(",") if e]
        if args.mul_netlist:
            # rename the netlist's own gl_reduce128 (different ports) and let its gl_mul override the RTL one
            net = os.path.join(work, "gl_mul_netlist.v")
            t = open(args.mul_netlist, encoding="utf-8", errors="replace").read()
            t = re.sub(r"\bgl_reduce128\b", "gl_reduce128_net", t)
            t = re.sub(r"(?s)module glbl \(\);.*?endmodule", "", t)
            with open(net, "w", newline="\n") as fh:
                fh.write(t)
            extra.append(net)
        if extra:
            cmds.append(f"{xv} " + " ".join(fwd(e) for e in extra))
        glbl = "<vivado-2022.2>/data/verilog/src/glbl.v"
        cmds.append(f"{xv} {glbl}")
        cmds.append(f"{xv} --sv -i {inc} {'-d NETLIST_DUT ' if args.dut_netlist else ''}{tb}")
        lib = "-L unisims_ver " if (args.mul_netlist or extra or args.dut_netlist) else ""
        cmds.append(f'{xe} -debug off {lib}-generic_top "LANES={args.lanes}" -generic_top "KCLK_MHZ={args.kclk}" '
                    f'-generic_top "LOG_NMAX={args.lognmax}" -s {snap} work.tb_axi_replay work.glbl')
        for i, c in enumerate(cmds):
            rc, out, el = sh(c, work, os.path.join(work, f"build_{i}.out"))
            rec["commands"].append(c)
            if rc != 0 or re.search(r"^ERROR", out, flags=re.M):
                print(out[-3000:])
                return 1
    c = (f'{xs} {snap} -R -testplusarg "TRACE={fwd(trace)}" -testplusarg "WMODE={args.wmode}" '
         f'-testplusarg "WDLY={args.wdly}" -testplusarg "GAP={args.gap}" -testplusarg "POWERON0={int(args.poweron0)}"')
    rc, out, el = sh(c, work, os.path.join(work, "xsim.out"))
    rec["commands"].append(c)
    rec["sim_seconds"] = round(el, 1)
    rds = re.findall(r"^RD (\w+) (\w+)\s*$", out, flags=re.M)
    done = re.search(r"REPLAY_DONE ops=(\d+) errors=(\d+) time_ns=(\d+)", out)
    rec["sim"] = {"reads": len(rds), "done": bool(done), "bus_errors": int(done.group(2)) if done else None,
                  "sim_time_ns": int(done.group(3)) if done else None}
    with open(values, "w", newline="\n") as fh:
        fh.write("\n".join(v for a, v in rds) + "\n")
    print("sim:", rec["sim"])

    # ------------------------------------------------------------ 3. playback + golden check
    if args.probe_trace:
        rec["read_values"] = [f"{a}:{v}" for a, v in rds]
    else:
        ptcl = os.path.join(work, "play_mode.tcl")
        with open(ptcl, "w", newline="\n") as fh:
            fh.write(make_tcl(args.tcl, "play", trace, values))
        plog = os.path.join(work, "play_mode.log")
        cmd = f'"{VIVADO}" -mode batch -nojournal -log "{plog}" -source "{ptcl}"'
        rc, out, el = sh(cmd, work, os.path.join(work, "play_mode.stdout"))
        rec["commands"].append(cmd)
        text = open(plog, encoding="utf-8", errors="replace").read() if os.path.exists(plog) else ""
        kv = jbt.parse_sfb(text)
        n = int(tcl_param(src_txt, "LOGN"))
        sfd = int(tcl_param(src_txt, "SEED_F"), 16)
        sgd = int(tcl_param(src_txt, "SEED_G"), 16)
        gold = sf.golden(sfd, sgd, n, args.lanes)
        chk = jbt.check(kv, gold, "auto", args.lanes)
        rec.update({"n": n, "seed_f": f"{sfd:016x}", "seed_g": f"{sgd:016x}",
                    "reads_used_by_playback": kv.get("REPLAY_READS_USED"),
                    "sfb": {k: v for k, v in kv.items() if k.startswith(("MSG_", "FIN_", "CYC_", "ID", "CFG", "STATUS", "ROUND"))},
                    **chk})
        print(json.dumps({"match": chk["match"], "n_errors": len(chk["errors"]), "first_errors": chk["errors"][:4]}, indent=1))
    out_path = args.out or os.path.join(ROOT, "results", "raw", "fpga_sim", f"replay_{args.tag}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="\n") as fh:
        json.dump(rec, fh, indent=1)
    print("wrote", out_path)
    return 0

if __name__ == "__main__":
    sys.exit(main())
