#!/usr/bin/env python3
"""Small on-board debug probes for the SumFold ZCU102 bring-up (JTAG-to-AXI, Vivado hw_manager).

Each probe is a short Vivado batch session (volatile PL configuration only; --run --authorized is required, exactly like fpga/host/jtag_board_test.py). The Tcl script is generated from a list of register operations, printed as SFP_<KEY>=<VALUE> lines and compared here with the golden model.

Probes:
  genread   : LOGN/seeds/GEN, then MEM read-back of selected f and g elements of every lane
              (compared with sumfold.gen_vector), then an AUTO job and all round messages / finals.
  memtab    : small tables written through the MEM port (patterns: ones, iota, mixed, gen), MEM
              read-back of every element, AUTO job, messages and lane finals vs sumfold.device_expected.

Examples: python fpga/host/board_probe.py genread --lanes 8 --logn 10 --bit fpga/build/L8_250MHz/sumfold_top.bit --run --authorized python fpga/host/board_probe.py memtab --lanes 8 --logn 4 --pattern ones --bit ... --no-program --run --authorized Add --sim to emit only the operation trace (for the xsim AXI replay testbench) without hardware.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from edgecert import sumfold as sf   # noqa: E402
from edgecert.field import P, ehex   # noqa: E402

GEN_DIR = os.path.join(HERE, "generated")
VIVADO = r"<vivado-2022.2>\bin\vivado.bat"
M64 = (1 << 64) - 1

TCL_HEAD = r"""
set BITSTREAM {@BIT@}
set PROBES    {@LTX@}
set PROGRAM   @PROGRAM@
proc kv {k v} { puts "SFP_$k=$v" }
proc hex32 {v} { return [format %08x [expr {$v & 0xFFFFFFFF}]] }
open_hw_manager
connect_hw_server -url localhost:3121 -allow_non_jtag
set tgt [lindex [get_hw_targets */xilinx_tcf/Digilent/210308B85974] 0]
if {$tgt eq ""} { kv ERROR "hw target not found"; exit 2 }
current_hw_target $tgt
open_hw_target
set dev [lindex [get_hw_devices xczu9_0] 0]
current_hw_device $dev
refresh_hw_device -update_hw_probes false $dev
if {$PROGRAM} {
    set_property PROGRAM.FILE $BITSTREAM $dev
    if {$PROBES ne ""} { set_property PROBES.FILE $PROBES $dev }
    program_hw_devices $dev
    kv PROGRAMMED 1
} else {
    if {$PROBES ne ""} { set_property PROBES.FILE $PROBES $dev }
}
refresh_hw_device $dev
set AXI [lindex [get_hw_axis hw_axi_1] 0]
if {$AXI eq ""} { kv ERROR "hw_axi_1 not found"; exit 3 }
reset_hw_axi $AXI
proc W {addr data} {
    global AXI
    create_hw_axi_txn -force sf_wr $AXI -address [format %08X $addr] \
        -data [format %08X [expr {$data & 0xFFFFFFFF}]] -len 1 -type write
    run_hw_axi -quiet [get_hw_axi_txns sf_wr]
}
proc R {addr} {
    global AXI
    create_hw_axi_txn -force sf_rd $AXI -address [format %08X $addr] -len 1 -type read
    run_hw_axi -quiet [get_hw_axi_txns sf_rd]
    return [expr "0x[get_property DATA [get_hw_axi_txns sf_rd]]"]
}
proc POLL {addr mask tag} {
    for {set i 0} {$i < 20000} {incr i} {
        set s [R $addr]
        if {$s & 0x4} { kv ERROR "STATUS.ERROR while waiting for $tag"; return $s }
        if {$s & $mask} { kv POLL_$tag [hex32 $s]; return $s }
    }
    kv ERROR "timeout $tag"
    return -1
}
proc SYSMON {tag} {
    if {[catch {
        set sm [lindex [get_hw_sysmons] 0]
        refresh_hw_sysmon $sm
        foreach prop {TEMPERATURE VCCINT VCCAUX VCCBRAM} {
            if {![catch {get_property $prop $sm} v]} { kv SYSMON_${tag}_$prop $v }
        }
    } err]} { kv SYSMON_${tag}_ERR $err }
}
SYSMON START
set T0 [clock milliseconds]
"""

TCL_TAIL = r"""
SYSMON END
kv ELAPSED_MS [expr {[clock milliseconds] - $T0}]
close_hw_target
disconnect_hw_server
close_hw_manager
kv END 1
"""

LINKRATE_TCL = r"""
proc TUS {} { return [clock microseconds] }
# A: one create_hw_axi_txn + run_hw_axi per single-beat write (the method of the board test script)
set t0 [TUS]
for {set i 0} {$i < @K@} {incr i} { W 0x0A8 $i }
set t1 [TUS]
kv LR_WRITE_CR_N @K@
kv LR_WRITE_CR_US [expr {$t1 - $t0}]
kv LR_WRITE_CR_LASTOK [expr {[R 0x0A8] == (@K@ - 1)}]
# B: create + run per single-beat read (ID register, value checked every time)
set bad 0
set t0 [TUS]
for {set i 0} {$i < @K@} {incr i} { if {[R 0x000] != 0xEDCE2026} { incr bad } }
set t1 [TUS]
kv LR_READ_CR_N @K@
kv LR_READ_CR_US [expr {$t1 - $t0}]
kv LR_READ_CR_BAD $bad
# C: write/read-back integrity on the readable data latches MEM_W0..W3
set bad 0
for {set i 0} {$i < 256} {incr i} {
    set a [expr {0x0A8 + 4 * ($i % 4)}]
    set v [expr {($i * 2654435761 + 12345) & 0xFFFFFFFF}]
    W $a $v
    if {[R $a] != $v} { incr bad }
}
kv LR_RWCHECK_N 256
kv LR_RWCHECK_BAD $bad
# D: pre-created transaction, run only
create_hw_axi_txn -force lr_w $AXI -address 000000A8 -data 5A5A1234 -len 1 -type write
set t0 [TUS]
for {set i 0} {$i < @K@} {incr i} { run_hw_axi -quiet [get_hw_axi_txns lr_w] }
set t1 [TUS]
kv LR_WRITE_RUN_N @K@
kv LR_WRITE_RUN_US [expr {$t1 - $t0}]
create_hw_axi_txn -force lr_r $AXI -address 00000000 -len 1 -type read
set bad 0
set t0 [TUS]
for {set i 0} {$i < @K@} {incr i} {
    run_hw_axi -quiet [get_hw_axi_txns lr_r]
    if {[string toupper [get_property DATA [get_hw_axi_txns lr_r]]] ne "EDCE2026"} { incr bad }
}
set t1 [TUS]
kv LR_READ_RUN_N @K@
kv LR_READ_RUN_US [expr {$t1 - $t0}]
kv LR_READ_RUN_BAD $bad
# E: queued single-beat transactions, @Q@ per run_hw_axi call
if {[catch {
    for {set q 0} {$q < @Q@} {incr q} {
        create_hw_axi_txn -force [format lrqw_%04d $q] $AXI -address [format %08X [expr {0x0A8 + 4 * ($q % 4)}]] \
            -data [format %08X [expr {$q + 0x100}]] -len 1 -type write
    }
    set nrep [expr {@K@ / @Q@}]
    set t0 [TUS]
    for {set r 0} {$r < $nrep} {incr r} { run_hw_axi -quiet [get_hw_axi_txns lrqw_*] }
    set t1 [TUS]
    kv LR_WRITE_Q_N [expr {$nrep * @Q@}]
    kv LR_WRITE_Q_US [expr {$t1 - $t0}]
    kv LR_WRITE_Q_CHECK [format %08X [R 0x0A8]]
} err]} { kv LR_WRITE_Q_ERR [string map {"\n" " "} $err] }
if {[catch {
    for {set q 0} {$q < @Q@} {incr q} {
        create_hw_axi_txn -force [format lrqr_%04d $q] $AXI -address 00000000 -len 1 -type read
    }
    set nrep [expr {@K@ / @Q@}]
    set t0 [TUS]
    for {set r 0} {$r < $nrep} {incr r} { run_hw_axi -quiet [get_hw_axi_txns lrqr_*] }
    set t1 [TUS]
    set bad 0
    foreach t [get_hw_axi_txns lrqr_*] { if {[string toupper [get_property DATA $t]] ne "EDCE2026"} { incr bad } }
    kv LR_READ_Q_N [expr {$nrep * @Q@}]
    kv LR_READ_Q_US [expr {$t1 - $t0}]
    kv LR_READ_Q_BAD $bad
} err]} { kv LR_READ_Q_ERR [string map {"\n" " "} $err] }
# F: burst (len 4) on the AXI4-Lite master
if {[catch {
    create_hw_axi_txn -force lr_b $AXI -address 000000A8 -data {11111111_22222222_33333333_44444444} -len 4 -type write
    run_hw_axi -quiet [get_hw_axi_txns lr_b]
    kv LR_BURST_W0 [format %08X [R 0x0A8]]
    kv LR_BURST_W1 [format %08X [R 0x0AC]]
    kv LR_BURST_W2 [format %08X [R 0x0B0]]
    kv LR_BURST_W3 [format %08X [R 0x0B4]]
    kv LR_BURST_OK 1
} err]} { kv LR_BURST_ERR [string map {"\n" " "} $err] }
# G: MEM-port element writes (MEM_ADDR, MEM_SEL, MEM_W0..W3, MEM_WE = 7 single-beat writes per
#    128-bit element) into f/g of a GEN-less table, then read-back of 16 of them
set M 128
set t0 [TUS]
for {set e 0} {$e < $M} {incr e} {
    W 0x0A0 [expr {$e >> 1}]
    W 0x0A4 [expr {$e & 1}]
    W 0x0A8 [expr {$e * 7 + 1}]
    W 0x0AC 0
    W 0x0B0 0
    W 0x0B4 0
    W 0x0B8 1
}
set t1 [TUS]
kv LR_MEMELEM_N $M
kv LR_MEMELEM_US [expr {$t1 - $t0}]
set bad 0
for {set e 0} {$e < $M} {incr e 8} {
    W 0x0A0 [expr {$e >> 1}]
    W 0x0A4 [expr {$e & 1}]
    W 0x0B8 2
    if {[R 0x0C0] != ($e * 7 + 1) || [R 0x0C4] != 0} { incr bad }
}
kv LR_MEMELEM_READBACK_N [expr {$M / 8}]
kv LR_MEMELEM_READBACK_BAD $bad
W 0x0A0 0
W 0x0A4 0
"""

SYSMON_LOOP_TCL = r"""
proc POLLQ {addr mask} {
    for {set i 0} {$i < 20000} {incr i} {
        set s [R $addr]
        if {$s & 0x4} { return -1 }
        if {$s & $mask} { return $s }
    }
    return -2
}
SYSMON PRE
set nbad 0
set tl0 [clock microseconds]
for {set j 0} {$j < @JOBS@} {incr j} {
    W 0x008 0x2
    if {[POLLQ 0x00C 0x10] < 0} { incr nbad }
    W 0x008 0x1
    if {[POLLQ 0x00C 0x2] < 0} { incr nbad }
    if {($j + 1) % @EVERY@ == 0} {
        kv LOOP_T_US_[expr {$j + 1}] [expr {[clock microseconds] - $tl0}]
        SYSMON J[expr {$j + 1}]
    }
}
kv LOOP_JOBS @JOBS@
kv LOOP_US [expr {[clock microseconds] - $tl0}]
kv LOOP_POLL_FAIL $nbad
SYSMON POST
"""

def device_with_challenges(f, g, lanes, chs):
    """Device outputs (AUTO) for arbitrary tables and arbitrary challenges r_0..r_{R-1}.

    Same arithmetic as sumfold.device_expected, but folding with the given challenges instead of the
    Fiat-Shamir ones (used for the negative tests, where the device is fed a flipped value)."""
    n = len(f).bit_length() - 1
    R = sf.device_rounds(n, lanes)
    cf, cg, msgs = list(f), list(g), []
    for i in range(R):
        msgs.append(sf.round_message(cf, cg))
        cf, cg = sf.fold(cf, chs[i]), sf.fold(cg, chs[i])
    return {"msgs": msgs, "lane_f": cf, "lane_g": cg}

def host_verify(n, lanes, claim, dev_msgs, lane_f, lane_g):
    """Honest host verifier: derives every challenge by Fiat-Shamir from the received messages,
    finishes the last log2(L) rounds from the lane residues and runs sumfold.verify.
    Returns (accepted, first_failure)."""
    R = sf.device_rounds(n, lanes)
    tr = sf.Transcript(n, claim)
    chs, c, first = [], claim, None
    for i, (s0, s1, s2) in enumerate(dev_msgs):
        if first is None and sf.eadd(s0, s1) != c:
            first = f"round {i}: s0 + s1 != running claim"
        r = tr.round(s0, s1, s2)
        chs.append(r)
        c = sf.interp_quadratic(s0, s1, s2, r)
    # tail challenges continue the same transcript over the host-computed messages
    cf, cg, tail = list(lane_f), list(lane_g), []
    for _ in range(n - R):
        m = sf.round_message(cf, cg)
        tail.append(m)
        r = tr.round(*m)
        chs.append(r)
        cf, cg = sf.fold(cf, r), sf.fold(cg, r)
    pf = sf.SumFoldProof(n=n, claim=claim, msgs=list(dev_msgs) + tail, challenges=chs,
                         final_f=cf[0], final_g=cg[0])
    ok = sf.verify(pf)
    if not ok and first is None:
        first = "final check or a host-tail round failed"
    return ok, first

# ------------------------------------------------------------------ op list -> Tcl / trace
class Ops:
    """Register operations. Reads carry a key; 128-bit reads are four 32-bit reads."""

    def __init__(self):
        self.ops = []   # (kind, addr, data_or_mask, key)
        self.pfx = ""   # key prefix (repetitions)

    def w(self, a, d):
        self.ops.append(("W", a, d & 0xFFFFFFFF, None))

    def r(self, a, key):
        self.ops.append(("R", a, 0, self.pfx + key))

    def poll(self, a, mask, key):
        self.ops.append(("P", a, mask, self.pfx + key))

    def raw(self, tcl):
        self.ops.append(("T", 0, 0, tcl))

    def w128(self, base, v):
        for k in range(4):
            self.w(base + 4 * k, (v >> (32 * k)) & 0xFFFFFFFF)

    def r128(self, base, key):
        for k in range(4):
            self.r(base + 4 * k, f"{key}.w{k}")

    def tcl(self) -> str:
        out = []
        for kind, a, d, key in self.ops:
            if kind == "W":
                out.append(f"W 0x{a:03X} 0x{d:08X}")
            elif kind == "R":
                out.append(f"kv {key} [hex32 [R 0x{a:03X}]]")
            elif kind == "T":
                out.append(key)
            else:
                out.append(f"POLL 0x{a:03X} 0x{d:X} {key}")
        return "\n".join(out) + "\n"

    def trace(self) -> str:
        """Trace for fpga/tb/tb_axi_replay.sv: W addr data | R addr | P addr mask (hex)."""
        out = []
        for kind, a, d, key in self.ops:
            if kind == "W":
                out.append(f"W {a:08x} {d:08x}")
            elif kind == "R":
                out.append(f"R {a:08x} 00000000")
            elif kind == "P":
                out.append(f"P {a:08x} {d:08x}")
        return "\n".join(out) + "\n"

    def read_keys(self):
        return [key for kind, a, d, key in self.ops if kind in ("R", "P")]

def assemble128(kv: dict, key: str):
    try:
        w = [int(kv[f"{key}.w{k}"], 16) for k in range(4)]
    except KeyError:
        return None
    return (w[1] << 32 | w[0], w[3] << 32 | w[2])

def ehex_t(e):
    return None if e is None else ehex(e)

# ------------------------------------------------------------------ probe builders
def common_setup(ops: Ops, n: int, seed_f: int, seed_g: int, gen: bool):
    ops.r(0x000, "ID")
    ops.r(0x004, "CFG")
    ops.w(0x008, 0x4)                      # soft reset, AUTO
    ops.w(0x010, n)
    ops.w(0x018, seed_f & 0xFFFFFFFF)
    ops.w(0x01C, seed_f >> 32)
    ops.w(0x020, seed_g & 0xFFFFFFFF)
    ops.w(0x024, seed_g >> 32)
    if gen:
        ops.w(0x008, 0x2)
        ops.poll(0x00C, 0x10, "GEN_DONE")
        ops.r(0x088, "CYC_GEN")

def mem_read(ops: Ops, idx: int, sel: int):
    ops.w(0x0A0, idx)
    ops.w(0x0A4, sel)
    ops.w(0x0B8, 2)
    ops.r128(0x0C0, f"MEM_{'fg'[sel]}_{idx}")
    ops.r(0x0D0, f"MEM_RCNT_{'fg'[sel]}_{idx}")

def mem_write(ops: Ops, idx: int, sel: int, e):
    ops.w(0x0A0, idx)
    ops.w(0x0A4, sel)
    ops.w128(0x0A8, e[0] | (e[1] << 64))
    ops.w(0x0B8, 1)

def job(ops: Ops, challenges, R: int, lanes: int):
    for i in range(R):
        ops.w(0x040, i)
        c = challenges[i]
        ops.w128(0x044, c[0] | (c[1] << 64))
        ops.w(0x054, 1)
    ops.w(0x008, 0x1)
    ops.poll(0x00C, 0x2, "DONE")
    ops.r(0x00C, "STATUS")
    ops.r(0x014, "ROUND")
    for i in range(R):
        for k in range(3):
            ops.w(0x060, 3 * i + k)
            ops.r128(0x064, f"MSG_{i}_{k}")
    for l in range(lanes):
        for k in range(2):
            ops.w(0x060, 48 + 2 * l + k)
            ops.r128(0x064, f"FIN_{'FG'[k]}_{l}")
    for i in range(R + 1):
        ops.w(0x090, i)
        ops.r(0x094, f"CYC_PASS_{i}")
    ops.r(0x080, "CYC_TOTAL")

def pattern_tables(pattern: str, N: int, seed_f: int, seed_g: int):
    if pattern == "ones":
        return [(1, 0)] * N, [(1, 0)] * N
    if pattern == "iota":
        return [(i, 0) for i in range(N)], [(1, 0)] * N
    if pattern == "iota2":
        return [(i + 1, 0) for i in range(N)], [(2 * i + 3, 0) for i in range(N)]
    if pattern == "big":
        return ([((P - 1 - i) % P, 0) for i in range(N)], [((1 << 63) + 12345 * i, 0) for i in range(N)])
    if pattern == "gen":
        return sf.gen_vector(seed_f, N), sf.gen_vector(seed_g, N)
    if pattern == "ext":
        import random
        rng = random.Random(7)
        return ([(rng.randrange(P), rng.randrange(P)) for _ in range(N)],
                [(rng.randrange(P), rng.randrange(P)) for _ in range(N)])
    raise SystemExit(f"unknown pattern {pattern}")

def build(args):
    L, n = args.lanes, args.logn
    N = 1 << n
    seed_f, seed_g = int(args.seed_f, 16), int(args.seed_g, 16)
    ops = Ops()
    expect = {}
    if args.probe == "genread":
        common_setup(ops, n, seed_f, seed_g, gen=True)
        f, g = sf.gen_vector(seed_f, N), sf.gen_vector(seed_g, N)
        per = N // L
        idxs = []
        for l in range(L):
            for k in sorted(set([0, 1, 2, 3, per - 2, per - 1])):
                if 0 <= k < per:
                    idxs.append(l * per + k)
        if args.all_elems:
            idxs = list(range(N))
        for i in idxs:
            for sel in (0, 1):
                mem_read(ops, i, sel)
                expect[f"MEM_{'fg'[sel]}_{i}"] = ehex((f, g)[sel][i])
    elif args.probe == "memtab":
        common_setup(ops, n, seed_f, seed_g, gen=False)
        f, g = pattern_tables(args.pattern, N, seed_f, seed_g)
        for i in range(N):
            mem_write(ops, i, 0, f[i])
            mem_write(ops, i, 1, g[i])
        for i in range(N):
            for sel in (0, 1):
                mem_read(ops, i, sel)
                expect[f"MEM_{'fg'[sel]}_{i}"] = ehex((f, g)[sel][i])
    elif args.probe == "rep":
        # the same GEN + AUTO job repeated: separates deterministic from non-deterministic errors
        f, g = sf.gen_vector(seed_f, N), sf.gen_vector(seed_g, N)
        d = sf.device_expected(f, g, L)
        R = d["device_rounds"]
        maddr = [int(x, 0) for x in args.mem_addr.split(",")] if args.mem_addr else []
        for rep in range(args.reps):
            ops.pfx = f"R{rep}_"
            common_setup(ops, n, seed_f, seed_g, gen=True)
            if maddr:
                # MEM_ADDR only selects the lane read address while no pass is running (idle reads)
                ops.w(0x0A0, maddr[rep % len(maddr)])
            job(ops, [sf.parse_ehex(h) for h in d["challenges"]], R, L)
            for i in range(R):
                for k in range(3):
                    expect[f"R{rep}_MSG_{i}_{k}"] = d["msgs"][i][k]
            for l in range(L):
                expect[f"R{rep}_FIN_F_{l}"] = d["lane_final_f"][l]
                expect[f"R{rep}_FIN_G_{l}"] = d["lane_final_g"][l]
        ops.pfx = ""
        return ops, expect, d
    elif args.probe == "linkrate":
                # JTAG-to-AXI link rate. Only harmless registers are touched: MEM_W0..W3 (data latches, no MEM_WE strobe, readable) and ID. AXI4-Lite master: every transaction is one 32-bit beat.
        K = args.count
        ops.r(0x000, "ID")
        ops.r(0x004, "CFG")
        ops.raw(LINKRATE_TCL.replace("@K@", str(K)).replace("@Q@", str(args.queue)))
        return ops, {}, None
    elif args.probe in ("negtable", "negch"):
                # negative test: honest positive control job, then the same job with one bit flipped either in a table element written through the MEM port (negtable) or in one challenge (negch).
        f, g = sf.gen_vector(seed_f, N), sf.gen_vector(seed_g, N)
        d = sf.device_expected(f, g, L)
        R = d["device_rounds"]
        chs = [sf.parse_ehex(h) for h in d["challenges"]]
        ops.pfx = "POS_"
        common_setup(ops, n, seed_f, seed_g, gen=True)
        job(ops, chs, R, L)
        for i in range(R):
            for k in range(3):
                expect[f"POS_MSG_{i}_{k}"] = d["msgs"][i][k]
        for l in range(L):
            expect[f"POS_FIN_F_{l}"] = d["lane_final_f"][l]
            expect[f"POS_FIN_G_{l}"] = d["lane_final_g"][l]
        ops.pfx = "NEG_"
        common_setup(ops, n, seed_f, seed_g, gen=True)
        f2, chs2 = list(f), list(chs)
        if args.probe == "negtable":
            j, b = args.flip_index, args.flip_bit
            f2[j] = (f[j][0] ^ (1 << b), f[j][1])
            mem_write(ops, j, 0, f2[j])
            mem_read(ops, j, 0)
            expect[f"NEG_MEM_f_{j}"] = ehex(f2[j])
            ops.w(0x0A0, 0)          # leave MEM_ADDR at its reset value
        else:
            j, b = args.flip_index, args.flip_bit
            chs2[j] = (chs[j][0] ^ (1 << b), chs[j][1])
        job(ops, chs2, R, L)
        dn = device_with_challenges(f2, g, L, chs2)
        for i in range(R):
            for k in range(3):
                expect[f"NEG_MSG_{i}_{k}"] = ehex(dn["msgs"][i][k])
        for l in range(L):
            expect[f"NEG_FIN_F_{l}"] = ehex(dn["lane_f"][l])
            expect[f"NEG_FIN_G_{l}"] = ehex(dn["lane_g"][l])
        ops.pfx = ""
        d["_neg"] = {"flip": {"target": "f table element via MEM port" if args.probe == "negtable" else "challenge",
                              "index": j, "bit": b, "component": "c0"},
                     "honest_claim": d["claim"], "f": f, "g": g}
        return ops, expect, d
    elif args.probe == "sysmon":
        # die temperature and rails before, while and after the kernel runs back-to-back jobs
        f, g = sf.gen_vector(seed_f, N), sf.gen_vector(seed_g, N)
        d = sf.device_expected(f, g, L)
        R = d["device_rounds"]
        common_setup(ops, n, seed_f, seed_g, gen=False)
        for i, c in enumerate(sf.parse_ehex(h) for h in d["challenges"]):
            ops.w(0x040, i)
            ops.w128(0x044, c[0] | (c[1] << 64))
            ops.w(0x054, 1)
        ops.raw(SYSMON_LOOP_TCL.replace("@JOBS@", str(args.count)).replace("@EVERY@", str(args.every)))
        # the last job's results must still be exact
        ops.r(0x00C, "STATUS")
        for i in range(R):
            for k in range(3):
                ops.w(0x060, 3 * i + k)
                ops.r128(0x064, f"MSG_{i}_{k}")
                expect[f"MSG_{i}_{k}"] = d["msgs"][i][k]
        for l in range(L):
            for k in range(2):
                ops.w(0x060, 48 + 2 * l + k)
                ops.r128(0x064, f"FIN_{'FG'[k]}_{l}")
            expect[f"FIN_F_{l}"] = d["lane_final_f"][l]
            expect[f"FIN_G_{l}"] = d["lane_final_g"][l]
        ops.r(0x080, "CYC_TOTAL")
        ops.r(0x088, "CYC_GEN")
        return ops, expect, d
    elif args.probe == "clk":
                # kernel clock: in STEP mode the core sits in S_WAIT and CYC_WALL counts every kernel clock cycle; two reads separated by a host sleep give f_kclk = dCYC / dt.
        ops.r(0x000, "ID")
        ops.r(0x004, "CFG")
        ops.w(0x008, 0x14)
        ops.w(0x010, n)
        ops.w(0x018, seed_f & 0xFFFFFFFF)
        ops.w(0x01C, seed_f >> 32)
        ops.w(0x020, seed_g & 0xFFFFFFFF)
        ops.w(0x024, seed_g >> 32)
        ops.w(0x008, 0x12)
        ops.poll(0x00C, 0x10, "GEN_DONE")
        ops.w(0x008, 0x11)
        ops.poll(0x00C, 0x8, "WAIT")
        ops.raw("set ta0 [clock microseconds]; set ca [R 0x098]; set cah [R 0x09C]; set ta1 [clock microseconds]\n"
                f"after {args.wait_ms}\n"
                "set tb0 [clock microseconds]; set cb [R 0x098]; set cbh [R 0x09C]; set tb1 [clock microseconds]\n"
                "kv CYC_A [expr {(wide($cah) << 32) | $ca}]; kv CYC_B [expr {(wide($cbh) << 32) | $cb}]\n"
                "kv T_A0 $ta0; kv T_A1 $ta1; kv T_B0 $tb0; kv T_B1 $tb1\n"
                "set dc [expr {((wide($cbh) << 32) | $cb) - ((wide($cah) << 32) | $ca)}]\n"
                "kv F_MHZ_MIN [format %.3f [expr {double($dc) / double($tb1 - $ta0)}]]\n"
                "kv F_MHZ_MAX [format %.3f [expr {double($dc) / double($tb0 - $ta1)}]]\n"
                "kv F_MHZ_MID [format %.3f [expr {double($dc) / double(($tb0 + $tb1) / 2 - ($ta0 + $ta1) / 2)}]]")
        ops.w(0x008, 0x4)
        ops.r(0x00C, "STATUS_AFTER_RESET")
        return ops, {}, None
    else:
        raise SystemExit("unknown probe")
    d = sf.device_expected(f, g, L)
    R = d["device_rounds"]
    if not args.no_job:
        job(ops, [sf.parse_ehex(h) for h in d["challenges"]], R, L)
        for i in range(R):
            for k in range(3):
                expect[f"MSG_{i}_{k}"] = d["msgs"][i][k]
        for l in range(L):
            expect[f"FIN_F_{l}"] = d["lane_final_f"][l]
            expect[f"FIN_G_{l}"] = d["lane_final_g"][l]
    return ops, expect, d

def compare(kv: dict, expect: dict):
    rows, bad = [], 0
    for key, exp in expect.items():
        got = ehex_t(assemble128(kv, key))
        ok = got == exp
        bad += not ok
        rows.append({"key": key, "got": got, "exp": exp, "ok": ok})
    return rows, bad

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probe", choices=("genread", "memtab", "clk", "rep", "linkrate", "negtable", "negch", "sysmon"))
    ap.add_argument("--count", type=int, default=1000, help="linkrate: transactions per method; sysmon: jobs")
    ap.add_argument("--queue", type=int, default=100, help="linkrate: transactions per run_hw_axi call")
    ap.add_argument("--every", type=int, default=50, help="sysmon: SYSMON sample every N jobs")
    ap.add_argument("--flip-index", type=int, default=5, help="negtable: f element index; negch: challenge index")
    ap.add_argument("--flip-bit", type=int, default=17, help="bit of the c0 word that is flipped")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--mem-addr", default="", help="rep: MEM_ADDR value(s) written before each job, comma list cycled over reps")
    ap.add_argument("--wait-ms", type=int, default=5000)
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--logn", type=int, default=10)
    ap.add_argument("--seed-f", default="0123456789abcdef")
    ap.add_argument("--seed-g", default="fedcba9876543210")
    ap.add_argument("--pattern", default="ones")
    ap.add_argument("--all-elems", action="store_true")
    ap.add_argument("--no-job", action="store_true")
    ap.add_argument("--bit", default="")
    ap.add_argument("--ltx", default="")
    ap.add_argument("--no-program", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--authorized", action="store_true")
    ap.add_argument("--trace", default="", help="write the op trace for tb_axi_replay.sv and exit")
    ap.add_argument("--replay-log", default="", help="compare an xsim replay log (RD lines) instead of hardware")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ops, expect, d = build(args)
    tag = args.tag or f"{args.probe}_{args.pattern if args.probe == 'memtab' else 'gen'}_L{args.lanes}_n{args.logn}"

    if args.trace:
        with open(args.trace, "w", newline="\n") as fh:
            fh.write(ops.trace())
        with open(args.trace + ".keys.json", "w", newline="\n") as fh:
            json.dump({"keys": ops.read_keys(), "expect": expect}, fh)
        print("trace", args.trace, "ops", len(ops.ops))
        return 0

    if args.replay_log:
        text = open(args.replay_log, encoding="utf-8", errors="replace").read()
        vals = re.findall(r"^RD (\w+) (\w+)$", text, flags=re.M)
        keys = ops.read_keys()
        kv = {k: v for k, (a, v) in zip(keys, vals)}
        rows, bad = compare(kv, expect)
        for r in rows:
            if not r["ok"]:
                print("MISMATCH", r["key"], "got", r["got"], "exp", r["exp"])
        print(json.dumps({"reads": len(vals), "expected_reads": len(keys), "checked": len(rows), "mismatches": bad}))
        return 0 if bad == 0 and len(vals) == len(keys) else 2

    if not (args.run and args.authorized):
        raise SystemExit("hardware probe needs --run --authorized (volatile JTAG programming approved by the author)")
    bit = os.path.abspath(args.bit).replace("\\", "/")
    ltx = os.path.abspath(args.ltx).replace("\\", "/") if args.ltx else ""
    tcl = (TCL_HEAD.replace("@BIT@", bit).replace("@LTX@", ltx).replace("@PROGRAM@", "0" if args.no_program else "1")
           + ops.tcl() + TCL_TAIL)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(GEN_DIR, exist_ok=True)
    tcl_path = os.path.join(GEN_DIR, f"probe_{tag}_{stamp}.tcl")
    log_path = os.path.join(GEN_DIR, f"probe_{tag}_{stamp}.log")
    with open(tcl_path, "w", newline="\n") as fh:
        fh.write(tcl)
    cmd = f'"{VIVADO}" -mode batch -nojournal -log "{log_path}" -source "{tcl_path}"'
    t0 = time.time()
    ts = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, errors="replace")
    el = time.time() - t0
    text = open(log_path, encoding="utf-8", errors="replace").read() if os.path.exists(log_path) else ""
    kv = dict(re.findall(r"^SFP_(\S+?)=(.*)$", text, flags=re.M))
    rows, bad = compare(kv, expect)
    sha = hashlib.sha256(open(args.bit, "rb").read()).hexdigest() if args.bit and os.path.exists(args.bit) else None
    rec = {
        "record": "sumfold_board_probe", "probe": args.probe, "pattern": args.pattern if args.probe == "memtab" else None,
        "status": "SUCCESS",
        "lanes": args.lanes, "n": args.logn, "seed_f": args.seed_f, "seed_g": args.seed_g,
        "bitstream": os.path.relpath(args.bit, ROOT).replace("\\", "/") if args.bit else None,
        "bitstream_sha256": sha, "programmed": not args.no_program,
        "timestamp": ts, "elapsed_s": round(el, 1),
        "invocation": "python " + " ".join([os.path.relpath(__file__, ROOT).replace("\\", "/")] + sys.argv[1:]),
        "command": cmd, "exit_code": p.returncode,
        "tcl": os.path.relpath(tcl_path, ROOT).replace("\\", "/"), "log": os.path.relpath(log_path, ROOT).replace("\\", "/"),
        "raw": {k: v for k, v in kv.items() if not re.match(r"(R\d+_|POS_|NEG_)?(MEM_|MSG_|FIN_)", k)},
        "checked": len(rows), "mismatches": bad, "match": bad == 0 and kv.get("END") == "1",
        "rows": rows,
    }
    if d is not None and "_neg" in d:
        neg = d["_neg"]
        claim = sf.parse_ehex(neg["honest_claim"])
        R = d["device_rounds"]
        out_v = {}
        for pfx in ("POS_", "NEG_"):
            try:
                msgs = [tuple(assemble128(kv, f"{pfx}MSG_{i}_{k}") for k in range(3)) for i in range(R)]
                lf = [assemble128(kv, f"{pfx}FIN_F_{l}") for l in range(args.lanes)]
                lg = [assemble128(kv, f"{pfx}FIN_G_{l}") for l in range(args.lanes)]
                if any(x is None for m in msgs for x in m) or None in lf or None in lg:
                    raise ValueError("missing words")
                acc, why = host_verify(args.logn, args.lanes, claim, msgs, lf, lg)
                out_v[pfx.rstrip("_").lower()] = {"host_verify_accepted": acc, "first_failure": why}
            except ValueError as e:
                out_v[pfx.rstrip("_").lower()] = {"host_verify_accepted": None, "first_failure": repr(e)}
        pos_rows = [r for r in rows if r["key"].startswith("POS_")]
        neg_rows = [r for r in rows if r["key"].startswith("NEG_")]
        rec["negative_test"] = {
            "flip": neg["flip"], "honest_claim": neg["honest_claim"],
            "positive_control": {**out_v["pos"], "device_equals_golden": all(r["ok"] for r in pos_rows),
                                 "words_checked": len(pos_rows)},
            "tampered": {**out_v["neg"],
                         "device_equals_golden_on_tampered_input": all(r["ok"] for r in neg_rows),
                         "words_checked": len(neg_rows)},
            "verifier": "honest host: Fiat-Shamir challenges re-derived from the received messages, host tail, "
                        "sumfold.verify against the honest claim H = <f, g> of the untampered tables",
        }
        rec["negative_test"]["pass"] = (rec["negative_test"]["positive_control"]["host_verify_accepted"] is True
                                        and rec["negative_test"]["tampered"]["host_verify_accepted"] is False)
    out = args.out or os.path.join(ROOT, "results", "raw", "fpga_board", "probes", f"probe_{tag}_{stamp}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="\n") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({k: rec[k] for k in ("probe", "lanes", "n", "checked", "mismatches", "match", "elapsed_s")}))
    print("raw", rec["raw"])
    for r in rows[:400]:
        if not r["ok"]:
            print("MISMATCH", r["key"], "got", r["got"], "exp", r["exp"])
    print("wrote", out)
    return 0

if __name__ == "__main__":
    sys.exit(main())
