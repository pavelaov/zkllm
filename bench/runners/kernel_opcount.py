"""Exact operation and memory-traffic counts for SumFold, fused vs unfused (ANALYTIC).

Writes results/raw/kernels/opcount.json with closed-form formulas in N = 2^n and their values for logN = 1..30, per full sumcheck (all n rounds, both tables f and g, Ext elements of 16 bytes).  The formulas are cross-checked against an instrumented model (counting array reads/writes and field operations while actually computing the transcript, which is also compared with the golden model) for logN = 1..12.

Schedules counted (spec/sumfold_kernel.md sec. 3.1):
  fused   : pass 0 evaluates level 0; pass i = 1..n-1 reads level i-1, writes level i in
            place and evaluates round i from the folded values; a final pass folds level
            n-1 -> level n (1 element).  n + 1 passes.
  unfused : per round an evaluate pass over level i and an out-of-place fold pass
            level i -> level i+1.  2n passes.
Both obtain the claim H as s0 + s1 of round 0 (no separate inner-product pass); the
golden prove() computes inner(f, g) separately, which would add N reads per table and
N Ext multiplications ('golden_literal_claim_pass').

Usage: python bench/runners/kernel_opcount.py [--out results/raw/kernels/opcount.json]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernel_common as kc  # noqa: E402

from edgecert import field as F  # noqa: E402
from edgecert.sumfold import Transcript, gen_vector  # noqa: E402

EXT_BYTES = 16
TABLES = 2

# ------------------------------------------------------------------ formulas
def formulas(n: int) -> dict:
    N = 1 << n
    pairs = N - 1                          # sum over rounds of N / 2^(i+1)
    ext_mul_eval = 3 * pairs               # f0g0, f1g1, (2f1-f0)(2g1-g0)
    ext_mul_fold = 2 * pairs               # r*(f1-f0), r*(g1-g0)
    ext_mul = ext_mul_eval + ext_mul_fold  # 5(N-1)
    d = {
        "N": N,
        "ext_mul": ext_mul,
        "base_mul_64x64": 3 * ext_mul,                 # Karatsuba: 15(N-1)
        "mul_by_7_shift_sub": ext_mul,                 # 5(N-1), multiplier-free
        "reduce128": 3 * ext_mul,                      # one per full product
                # base-field add/sub: eval pair = 2 dbl + 2 sub + 3 acc (Ext) = 14, + 3x5 Karatsuba = 29; fold pair (both tables) = 2 sub + 2 add (Ext) = 8, + 2x5 Karatsuba = 18
        "base_addsub": 47 * pairs,
        "sha256_compressions": 1 + 2 * n,              # 40-byte init block + n x 80-byte round inputs
        "fused": {
            "passes": n + 1,
            "reads_elems_per_table": 3 * N - 2,        # level 0 twice, levels 1..n-1 once
            "writes_elems_per_table": N - 1,           # levels 1..n
        },
        "unfused": {
            "passes": 2 * n,
            "reads_elems_per_table": 4 * N - 4,        # every level 0..n-1 twice
            "writes_elems_per_table": N - 1,
        },
    }
    for v in ("fused", "unfused"):
        x = d[v]
        x["bytes_read"] = TABLES * EXT_BYTES * x["reads_elems_per_table"]
        x["bytes_written"] = TABLES * EXT_BYTES * x["writes_elems_per_table"]
        x["bytes_total"] = x["bytes_read"] + x["bytes_written"]
        x["bytes_per_index"] = x["bytes_total"] / N          # index b <-> (f[b], g[b]) = 32 B of input
        x["bytes_per_table_element"] = x["bytes_total"] / (TABLES * N)
        x["base_mul_per_byte"] = d["base_mul_64x64"] / x["bytes_total"]
    d["fused"]["resident_bytes"] = TABLES * EXT_BYTES * N                # in place: exactly 2N x 128 bit
    d["unfused"]["resident_bytes"] = TABLES * EXT_BYTES * (N + max(N // 2, 1))  # ping-pong A[N] + B[N/2]
    d["gpu_pingpong_resident_bytes"] = d["unfused"]["resident_bytes"]
    d["base_mul_per_index"] = d["base_mul_64x64"] / N
    d["traffic_ratio_unfused_over_fused"] = d["unfused"]["bytes_total"] / d["fused"]["bytes_total"]
    return d

FORMULAS_TEXT = {
    "ext_mul": "5(N-1)  [eval 3(N-1) + fold 2(N-1)]",
    "base_mul_64x64": "15(N-1)",
    "mul_by_7_shift_sub": "5(N-1)",
    "reduce128": "15(N-1)",
    "base_addsub": "47(N-1)",
    "sha256_compressions": "1 + 2n",
    "fused.passes": "n + 1",
    "fused.reads_elems_per_table": "3N - 2",
    "fused.writes_elems_per_table": "N - 1",
    "fused.bytes_total": "2 tables x 16 B x (4N - 3) = 128N - 96",
    "unfused.passes": "2n",
    "unfused.reads_elems_per_table": "4N - 4",
    "unfused.writes_elems_per_table": "N - 1",
    "unfused.bytes_total": "2 tables x 16 B x (5N - 5) = 160N - 160",
    "fused.resident_bytes": "32N  (in place)",
    "unfused.resident_bytes": "48N  (ping-pong A[N] + B[N/2]; also the GPU kernels)",
    "asymptotic": "fused 128 B, unfused 160 B moved per index b (32 B of input); both 15 base "
                  "multiplications per index; unfused/fused traffic -> 1.25",
    "lane_overhead_fused_L_lanes": "CPU/RTL lane model with L lanes adds per table (2L - 1) reads and "
                                   "(L - 1) writes (gather of lane residues + re-read of level R in the host "
                                   "tail); passes become (R + 1) + (log2 L + 1) with R = n - log2 L",
    "gpu_round_overhead": "per evaluating pass: grid x 48 B partials written and read on device, "
                          "48 B device->host, 1 host SHA-256 round (2 compressions)",
    "golden_literal_claim_pass": "+N reads per table (+32N B) and +N Ext mults (+3N base mults) if H is "
                                 "computed by a separate inner-product pass as in the golden prove()",
    "base_input_specialization": "not used by the kernels (uniform 128-bit datapath, as in the RTL): with "
                                 "c1 = 0 inputs round 0 could use 3 instead of 9 base mults per pair for "
                                 "evaluation and 2 instead of 3 per table fold: total 11N - 15 base mults",
}

# ------------------------------------------------------- instrumented model
class CountingTable:
    def __init__(self, data, stats):
        self.d = list(data)
        self.s = stats

    def __getitem__(self, i):
        self.s["reads"] += 1
        return self.d[i]

    def __setitem__(self, i, v):
        self.s["writes"] += 1
        self.d[i] = v

class Ops:
    def __init__(self):
        self.c = {"ext_mul": 0, "base_addsub": 0}

    def mul(self, a, b):
        self.c["ext_mul"] += 1
        self.c["base_addsub"] += 5
        return F.emul(a, b)

    def add(self, a, b):
        self.c["base_addsub"] += 2
        return F.eadd(a, b)

    def sub(self, a, b):
        self.c["base_addsub"] += 2
        return F.esub(a, b)

def _eval(O, acc, f0, f1, g0, g1):
    p0, p1 = O.mul(f0, g0), O.mul(f1, g1)
    p2 = O.mul(O.sub(O.add(f1, f1), f0), O.sub(O.add(g1, g1), g0))
    return [O.add(acc[0], p0), O.add(acc[1], p1), O.add(acc[2], p2)]

def _fold(O, v0, v1, r):
    return O.add(v0, O.mul(r, O.sub(v1, v0)))

def instrumented(n: int, variant: str, seedf=kc.DEFAULT_SEED_F, seedg=kc.DEFAULT_SEED_G) -> dict:
    N = 1 << n
    st = {"reads": 0, "writes": 0}
    O = Ops()
    f = CountingTable(gen_vector(seedf, N), st)
    g = CountingTable(gen_vector(seedg, N), st)
    passes = 0
    msgs, chal = [], []
    tr = None
    if variant == "fused":
        acc = [F.ZERO] * 3
        for j in range(N // 2):
            acc = _eval(O, acc, f[2 * j], f[2 * j + 1], g[2 * j], g[2 * j + 1])
        passes += 1
        claim = F.eadd(acc[0], acc[1])
        tr = Transcript(n, claim)
        r = tr.round(*acc)
        msgs.append(tuple(acc)); chal.append(r)
        M = N
        for _ in range(1, n):
            acc = [F.ZERO] * 3
            for k in range(M // 4):
                a0 = _fold(O, f[4 * k], f[4 * k + 1], r)
                a1 = _fold(O, f[4 * k + 2], f[4 * k + 3], r)
                b0 = _fold(O, g[4 * k], g[4 * k + 1], r)
                b1 = _fold(O, g[4 * k + 2], g[4 * k + 3], r)
                f[2 * k], f[2 * k + 1], g[2 * k], g[2 * k + 1] = a0, a1, b0, b1
                acc = _eval(O, acc, a0, a1, b0, b1)
            passes += 1
            r = tr.round(*acc)
            msgs.append(tuple(acc)); chal.append(r)
            M //= 2
        f[0] = _fold(O, f[0], f[1], r)
        g[0] = _fold(O, g[0], g[1], r)
        passes += 1
        ff, gg = f.d[0], g.d[0]
    else:
        src_f, src_g = f, g
        M = N
        for i in range(n):
            acc = [F.ZERO] * 3
            for j in range(M // 2):
                acc = _eval(O, acc, src_f[2 * j], src_f[2 * j + 1], src_g[2 * j], src_g[2 * j + 1])
            passes += 1
            if i == 0:
                tr = Transcript(n, F.eadd(acc[0], acc[1]))
            r = tr.round(*acc)
            msgs.append(tuple(acc)); chal.append(r)
            dst_f = CountingTable([F.ZERO] * (M // 2), st)
            dst_g = CountingTable([F.ZERO] * (M // 2), st)
            for j in range(M // 2):
                dst_f[j] = _fold(O, src_f[2 * j], src_f[2 * j + 1], r)
                dst_g[j] = _fold(O, src_g[2 * j], src_g[2 * j + 1], r)
            passes += 1
            src_f, src_g = dst_f, dst_g
            M //= 2
        ff, gg = src_f.d[0], src_g.d[0]
    gold = kc.golden_record(seedf, seedg, n)
    out = {"claim": F.ehex(F.eadd(msgs[0][0], msgs[0][1])), "msgs": [[F.ehex(x) for x in m] for m in msgs],
           "challenges": [F.ehex(x) for x in chal], "final_f": F.ehex(ff), "final_g": F.ehex(gg)}
    return {"passes": passes, "reads": st["reads"], "writes": st["writes"], "ext_mul": O.c["ext_mul"],
            "base_addsub": O.c["base_addsub"], "match": kc.compare_records(out, gold) == []}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(kc.ROOT / "results" / "raw" / "kernels" / "opcount.json"))
    ap.add_argument("--check-max-n", type=int, default=12)
    args = ap.parse_args(argv)

    checks = []
    ok = True
    for n in range(1, args.check_max_n + 1):
        fm = formulas(n)
        for v in ("fused", "unfused"):
            ins = instrumented(n, v)
            exp = {"passes": fm[v]["passes"], "reads": 2 * fm[v]["reads_elems_per_table"],
                   "writes": 2 * fm[v]["writes_elems_per_table"], "ext_mul": fm["ext_mul"],
                   "base_addsub": fm["base_addsub"]}
            agree = all(ins[k] == exp[k] for k in exp) and ins["match"]
            ok &= agree
            checks.append({"n": n, "variant": v, "instrumented": ins, "formula": exp, "agree": agree})
    rec = {
        "status": "ANALYTIC_VERIFIED" if ok else "ANALYTIC_MISMATCH",
        "evidence_class": "ANALYTIC",
        "command": subprocess.list2cmdline([sys.executable] + sys.argv),
        "timestamp_utc": kc.now_utc(),
        "host": {"python": sys.version.split()[0]},
        "definitions": {
            "N": "2^n evaluations per table; two tables f, g",
            "element": "one Ext2 value, 16 bytes {c0, c1}",
            "index": "one b in {0,1}^n, i.e. the input pair (f[b], g[b]) = 32 bytes",
            "base_mul_64x64": "64x64->128-bit products (Karatsuba: 3 per Ext multiplication)",
            "traffic": "element loads/stores of the table arrays by the passes; excludes the 48-byte "
                       "round messages, partial sums, challenges and the transcript",
        },
        "formulas": FORMULAS_TEXT,
        "values": {str(n): formulas(n) for n in range(1, 31)},
        "instrumented_check": {"max_n": args.check_max_n, "all_agree": ok, "cases": checks},
        "notes": ["Arithmetic is identical in both variants; the fused pass removes one read of every "
                  "level >= 1 (N - 2 elements per table) and n - 1 passes.",
                  "Counts are protocol-level; the CPU kernel's lazy 128-bit accumulators replace the "
                  "3 Ext accumulations per pair by 128-bit integer additions (same result)."],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    v24 = formulas(24)
    print(json.dumps({"out": str(out), "status": rec["status"],
                      "n24_fused_bytes": v24["fused"]["bytes_total"], "n24_unfused_bytes": v24["unfused"]["bytes_total"],
                      "n24_base_mul": v24["base_mul_64x64"]}))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
