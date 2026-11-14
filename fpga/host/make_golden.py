#!/usr/bin/env python3
"""Golden vectors for the SumFold RTL testbenches (spec/sumfold_kernel.md).

Everything is computed by the read-only golden model (src/edgecert/sumfold.py, field.py). Outputs (default directory fpga/sim/golden/):

<case>.json        full golden dict (sumfold.golden / sumfold.device_expected) <case>_ch.hex      R challenges r_0..r_{R-1}, one 128-bit word {c1,c0} per line <case>_msg.hex     3R round-message words s0,s1,s2 of rounds 0..R-1 <case>_fin.hex     2L lane finals: lane0 f, lane0 g, lane1 f, lane1 g, ... <case>_f.hex/_g.hex  input tables (N words), used for MEM-write cases and GEN read-back checks cases_L<L>.txt     "name n mem step seed_f seed_g R" per case (read by tb_core.sv) field_*.hex        random/edge vectors for tb_field.sv manifest.json      list of cases

Usage:  python fpga/host/make_golden.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from edgecert import field as F      # noqa: E402
from edgecert import sumfold as sf   # noqa: E402

P = F.P
M64 = (1 << 64) - 1
NV_FIELD = 512

EDGE = [0, 1, 2, 3, 7, P - 1, P - 2, P - 7, (1 << 32) - 1, 1 << 32, (1 << 32) + 1,
        (1 << 63) - 1, 1 << 63, P - (1 << 32), (P - 1) // 2, (P + 1) // 2, 0xFFFFFFFF00000000]

# ------------------------------------------------------------------ helpers
def xorshift_inv(y: int) -> int:
    """Inverse of sf.xorshift64 (used to pick seeds whose first output is >= p)."""
    x = y & M64
    # undo x ^= x << 17
    t = x
    for s in (17, 34, 51):
        t ^= (x << s) & M64
    x = t
    # undo x ^= x >> 7
    t = x
    for s in range(7, 64, 7):
        t ^= x >> s
    x = t
    # undo x ^= x << 13
    t = x
    for s in (13, 26, 39, 52):
        t ^= (x << s) & M64
    x = t
    assert sf.xorshift64(x) == (y & M64)
    return x

def w128(e) -> str:
    return F.ehex(e)

def write_lines(path: str, lines) -> None:
    with open(path, "w", newline="\n") as fh:
        for ln in lines:
            fh.write(ln + "\n")

def rand_elem(rng: random.Random) -> int:
    return rng.choice(EDGE) if rng.random() < 0.5 else rng.randrange(P)

def table(kind: str, N: int, rng: random.Random):
    if kind == "zero":
        return [(0, 0)] * N
    if kind == "pm1":
        return [(P - 1, P - 1)] * N
    if kind == "pm1base":
        return [(P - 1, 0)] * N
    if kind == "mixed":
        return [(rand_elem(rng), rand_elem(rng)) for _ in range(N)]
    raise ValueError(kind)

# ------------------------------------------------------------------ cases
def emit_case(out: str, name: str, L: int, n: int, mem: bool, step: bool,
              seed_f: int, seed_g: int, f, g) -> dict:
    d = sf.device_expected(f, g, L)
    d.update({"seed_f": f"{seed_f:016x}", "seed_g": f"{seed_g:016x}"})
    R = d["device_rounds"]
    # consistency with the published golden() entry point for generator cases
    if not mem:
        gd = sf.golden(seed_f, seed_g, n, L)
        assert gd["msgs"] == d["msgs"] and gd["lane_final_f"] == d["lane_final_f"]
    write_lines(os.path.join(out, f"{name}_ch.hex"), d["challenges"])
    write_lines(os.path.join(out, f"{name}_msg.hex"), [s for m in d["msgs"] for s in m])
    fin = []
    for lf, lg in zip(d["lane_final_f"], d["lane_final_g"]):
        fin += [lf, lg]
    write_lines(os.path.join(out, f"{name}_fin.hex"), fin)
    write_lines(os.path.join(out, f"{name}_f.hex"), [w128(e) for e in f])
    write_lines(os.path.join(out, f"{name}_g.hex"), [w128(e) for e in g])
    rec = {"name": name, "lanes": L, "n": n, "mem": int(mem), "step": int(step),
           "seed_f": f"{seed_f:016x}", "seed_g": f"{seed_g:016x}", "device_rounds": R}
    with open(os.path.join(out, f"{name}.json"), "w", newline="\n") as fh:
        json.dump({**rec, "golden": d}, fh, indent=1)
    return rec

def build_cases(out: str) -> list:
    rng = random.Random(20260927)
    cases = []

    def gen_case(name, L, n, sfd, sgd, step=False):
        f = sf.gen_vector(sfd, 1 << n)
        g = sf.gen_vector(sgd, 1 << n)
        cases.append(emit_case(out, name, L, n, False, step, sfd, sgd, f, g))

    def mem_case(name, L, n, kind, step=False):
        N = 1 << n
        f = table(kind, N, rng)
        g = table(kind, N, rng)
        cases.append(emit_case(out, name, L, n, True, step, 1, 2, f, g))

    # generator (on-chip xorshift) cases, AUTO mode
    gen_case("g_L1_n4", 1, 4, 0x0123456789ABCDEF, 0xFEDCBA9876543210)
    gen_case("g_L1_n6", 1, 6, 0x1234567890ABCDEF, 0x0F1E2D3C4B5A6978)
    gen_case("g_L1_n8", 1, 8, 0xDEADBEEFCAFEF00D, 0x8BADF00D5EEDBEEF)
    gen_case("g_L2_n6", 2, 6, 0x243F6A8885A308D3, 0x13198A2E03707344)
    gen_case("g_L2_n8", 2, 8, 0xA4093822299F31D0, 0x082EFA98EC4E6C89)
    gen_case("g_L4_n8", 4, 8, 0x452821E638D01377, 0xBE5466CF34E90C6C)
    gen_case("g_L4_n10", 4, 10, 0xC0AC29B7C97C50DD, 0x3F84D5B5B5470917)
    gen_case("g_L8_n10", 8, 10, 0x9216D5D98979FB1B, 0xD1310BA698DFB5AC)
    # generator edge: first outputs >= p exercise the on-chip canonicalisation
    gen_case("g_L2_n5_canon", 2, 5, xorshift_inv(P + 5), xorshift_inv(M64))
    # minimum device rounds (R = 1)
    gen_case("g_L8_n4_r1", 8, 4, 0x2FFD72DBD01ADFB7, 0xB8E1AFED6A267E96)
    # STEP mode: testbench plays the host (reads each message, supplies the next challenge)
    gen_case("s_L2_n6_step", 2, 6, 0x0801F2E2858EFC16, 0x636920D871574E69, step=True)
    gen_case("s_L8_n6_step", 8, 6, 0xA458FEA3F4933D7E, 0x0D95748F728EB658, step=True)
    # full-size cases (n = LOG_NMAX = 16, every bank address used) for the exact per-pass cycle counts
    gen_case("g_L1_n16_full", 1, 16, 0x6A09E667F3BCC908, 0xBB67AE8584CAA73B)
    gen_case("g_L2_n16_full", 2, 16, 0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1)
    gen_case("g_L4_n16_full", 4, 16, 0x510E527FADE682D1, 0x9B05688C2B3E6C1F)
    gen_case("g_L8_n16_full", 8, 16, 0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179)
    # direct MEM-write cases with edge values (full extension elements, c1 != 0)
    mem_case("m_L1_n6_mixed", 1, 6, "mixed")
    mem_case("m_L1_n4_pm1", 1, 4, "pm1")
    mem_case("m_L1_n1_mixed", 1, 1, "mixed")
    mem_case("m_L1_n5_pm1base", 1, 5, "pm1base")
    mem_case("m_L2_n6_zero", 2, 6, "zero")
    mem_case("m_L2_n6_pm1", 2, 6, "pm1")
    mem_case("m_L2_n2_mixed", 2, 2, "mixed")
    mem_case("m_L4_n8_mixed", 4, 8, "mixed")
    mem_case("m_L8_n6_mixed", 8, 6, "mixed")
    mem_case("m_L8_n4_pm1", 8, 4, "pm1")
    mem_case("s_L1_n5_mixed_step", 1, 5, "mixed", step=True)
    return cases

# ------------------------------------------------------------------ field vectors
def field_vectors(out: str) -> None:
    rng = random.Random(7)

    def r64():
        return rng.choice(EDGE) if rng.random() < 0.3 else rng.randrange(P)

    add, sub, mul, m7, red, emul = [], [], [], [], [], []
    for i in range(NV_FIELD):
        a, b = r64(), r64()
        add.append(f"{a:016x}{b:016x}{F.fadd(a, b):016x}")
        sub.append(f"{a:016x}{b:016x}{F.fsub(a, b):016x}")
        mul.append(f"{a:016x}{b:016x}{F.fmul(a, b):016x}")
        m7.append(f"{a:016x}{F.fmul(7, a):016x}")
        if i < 16:
            x = [0, (1 << 128) - 1, P * P - 1, (P - 1) * (P - 1), 1 << 127, (1 << 96) - 1,
                 1 << 96, (1 << 64) - 1, 1 << 64, P, P - 1, (1 << 128) - (1 << 96),
                 ((1 << 32) - 1) << 64, (P << 64) - 1, 0xFFFFFFFF << 96, 12345][i]
        else:
            x = rng.randrange(1 << 128)
        red.append(f"{x:032x}{F.reduce128(x):016x}")
        ea, eb = (r64(), r64()), (r64(), r64())
        emul.append(F.ehex(ea) + F.ehex(eb) + F.ehex(F.emul(ea, eb)))
    for nm, lines in (("add", add), ("sub", sub), ("mul", mul), ("mul7", m7),
                      ("red", red), ("emul", emul)):
        write_lines(os.path.join(out, f"field_{nm}.hex"), lines)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "fpga", "sim", "golden"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cases = build_cases(args.out)
    for L in (1, 2, 4, 8):
        rows = [c for c in cases if c["lanes"] == L]
        write_lines(os.path.join(args.out, f"cases_L{L}.txt"),
                    [f'{c["name"]} {c["n"]} {c["mem"]} {c["step"]} {c["seed_f"]} {c["seed_g"]} {c["device_rounds"]}'
                     for c in rows])
    field_vectors(args.out)
    with open(os.path.join(args.out, "manifest.json"), "w", newline="\n") as fh:
        json.dump({"generator": "fpga/host/make_golden.py", "golden_model": "src/edgecert/sumfold.py",
                   "cases": cases, "field_vectors": NV_FIELD}, fh, indent=1)
    print(f"wrote {len(cases)} cases and field vectors to {args.out}")

if __name__ == "__main__":
    main()
