#!/usr/bin/env python3
"""SumFold on-board test driver for the ZCU102 (JTAG-to-AXI, Vivado hw_manager).

1. Computes the golden values with edgecert.sumfold.golden(seed_f, seed_g, n, L).
2. Renders fpga/scripts/board_test_template.tcl into a concrete Vivado Tcl script
   (fpga/host/generated/board_test_<tag>.tcl) that programs the bitstream, drives the register
   map through hw_axi_1 and prints SFB_<KEY>=<VALUE> lines.
     AUTO : challenges r_0..r_{R-1} are precomputed here and written before START.
     STEP : no challenge is precomputed; the Tcl script reads each round message from the board and
            derives r_i with the SHA-256 transcript of spec section 3 (tcllib sha256 in Vivado Tcl).
3. Parses the Vivado log, checks every message, lane final (and STEP challenge) against the golden
   model, finishes the last log2(L) rounds on the host (sumfold.host_tail) and runs sumfold.verify
   on the assembled transcript, then writes a result JSON.

Default is a DRY RUN: the hardware script is only generated; a second copy with DRY_RUN=1 is run in Vivado batch mode against a register emulator that answers with golden values (no hardware manager is opened, nothing is programmed). This validates the Tcl flow and the Tcl transcript. Executing on the board requires --run --authorized (volatile PL configuration via JTAG only).

Examples: python fpga/host/jtag_board_test.py --lanes 8 --logn 16 --mode auto  --bit build/sumfold_top.bit python fpga/host/jtag_board_test.py --lanes 8 --logn 12 --mode step  --bit build/sumfold_top.bit python fpga/host/jtag_board_test.py ... --run --authorized          (hardware; needs approval)
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

TEMPLATE = os.path.join(ROOT, "fpga", "scripts", "board_test_template.tcl")
GEN_DIR = os.path.join(HERE, "generated")
VIVADO = r"<vivado-2022.2>\bin\vivado.bat"

def fwd(p: str) -> str:
    return p.replace("\\", "/")

def render(subs: dict) -> str:
    txt = open(TEMPLATE, encoding="utf-8").read()
    for k, v in subs.items():
        txt = txt.replace(f"@@{k}@@", str(v))
    left = re.findall(r"@@\w+@@", txt)
    if left:
        raise SystemExit(f"unfilled placeholders: {sorted(set(left))}")
    return txt

def parse_sfb(text: str) -> dict:
    kv = {}
    for m in re.finditer(r"^SFB_(\w+)=(.*)$", text, flags=re.M):
        kv[m.group(1)] = m.group(2).strip()
    return kv

def check(kv: dict, gold: dict, mode: str, lanes: int) -> dict:
    R = gold["device_rounds"]
    errors = []
    for i in range(R):
        for k in range(3):
            got = kv.get(f"MSG_{i}_{k}")
            if got != gold["msgs"][i][k]:
                errors.append(f"msg[{i}][{k}] got {got} exp {gold['msgs'][i][k]}")
    for l in range(lanes):
        if kv.get(f"FIN_F_{l}") != gold["lane_final_f"][l]:
            errors.append(f"lane {l} final f got {kv.get(f'FIN_F_{l}')} exp {gold['lane_final_f'][l]}")
        if kv.get(f"FIN_G_{l}") != gold["lane_final_g"][l]:
            errors.append(f"lane {l} final g got {kv.get(f'FIN_G_{l}')} exp {gold['lane_final_g'][l]}")
    if mode == "step":
        if kv.get("CLAIM") != gold["claim"]:
            errors.append(f"claim got {kv.get('CLAIM')} exp {gold['claim']}")
        for i in range(R):
            if kv.get(f"CH_{i}") != gold["challenges"][i]:
                errors.append(f"challenge {i} (Tcl transcript) got {kv.get(f'CH_{i}')} exp {gold['challenges'][i]}")
    if "ERROR" in kv:
        errors.append("script error: " + kv["ERROR"])
    if kv.get("END") != "1":
        errors.append("script did not reach END")

    # host tail + transparent (non-ZK) verification of the assembled transcript
    verified = None
    try:
        dev_msgs = [[sf.parse_ehex(kv[f"MSG_{i}_{k}"]) for k in range(3)] for i in range(R)]
        lf = [sf.parse_ehex(kv[f"FIN_F_{l}"]) for l in range(lanes)]
        lg = [sf.parse_ehex(kv[f"FIN_G_{l}"]) for l in range(lanes)]
        all_ch = [sf.parse_ehex(h) for h in gold["all_challenges"]]
        tail_msgs, ff, gg = sf.host_tail(lf, lg, all_ch[R:])
        pf = sf.SumFoldProof(n=gold["n"], claim=sf.parse_ehex(gold["claim"]),
                             msgs=[tuple(m) for m in dev_msgs] + list(tail_msgs),
                             challenges=all_ch, final_f=ff, final_g=gg)
        verified = sf.verify(pf)
        if not verified:
            errors.append("sumfold.verify rejected the assembled transcript")
    except (KeyError, ValueError) as e:
        errors.append(f"cannot assemble transcript: {e!r}")
    return {"match": not errors, "errors": errors, "transcript_verified": verified}

def run_vivado(tcl: str, log: str) -> tuple[int, str, float]:
    cmd = f'"{VIVADO}" -mode batch -nojournal -log "{log}" -source "{tcl}"'
    t0 = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, errors="replace")
    return p.returncode, cmd, time.time() - t0

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lanes", type=int, default=8, choices=(1, 2, 4, 8))
    ap.add_argument("--logn", type=int, default=12)
    ap.add_argument("--seed-f", default="0123456789abcdef")
    ap.add_argument("--seed-g", default="fedcba9876543210")
    ap.add_argument("--mode", choices=("auto", "step"), default="auto")
    ap.add_argument("--bit", default=fwd(os.path.join(ROOT, "fpga", "build", "sumfold_top.bit")))
    ap.add_argument("--ltx", default="")
    ap.add_argument("--hw-server", default="localhost:3121")
    ap.add_argument("--hw-target", default="*/xilinx_tcf/Digilent/210308B85974")
    ap.add_argument("--hw-device", default="xczu9_0")
    ap.add_argument("--no-program", action="store_true", help="assume the bitstream is already loaded")
    ap.add_argument("--poll-max", type=int, default=20000)
    ap.add_argument("--run", action="store_true", help="execute on hardware (requires --authorized)")
    ap.add_argument("--authorized", action="store_true", help="author approved volatile JTAG programming")
    ap.add_argument("--no-vivado", action="store_true", help="dry run: only generate the scripts")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="", help="suffix for the generated Tcl/log names (campaign runs)")
    args = ap.parse_args()

    if args.run and not args.authorized:
        raise SystemExit("--run programs the ZCU102 PL over JTAG; pass --authorized only after author approval")

    seed_f, seed_g = int(args.seed_f, 16), int(args.seed_g, 16)
    L, n = args.lanes, args.logn
    t_gold = time.time()
    gold = sf.golden(seed_f, seed_g, n, L)
    t_gold = time.time() - t_gold
    R = gold["device_rounds"]
    tag = f"{args.mode}_L{L}_n{n}" + (f"_{args.tag}" if args.tag else "")
    os.makedirs(GEN_DIR, exist_ok=True)

    common = {
        "BITSTREAM": fwd(os.path.abspath(args.bit)), "PROBES": fwd(args.ltx), "HW_SERVER": args.hw_server,
        "HW_TARGET": args.hw_target, "HW_DEVICE": args.hw_device, "PROGRAM": 0 if args.no_program else 1,
        "LOGN": n, "LANES": L, "SEED_F": f"{seed_f:016x}", "SEED_G": f"{seed_g:016x}", "MODE": args.mode,
        # STEP must derive every challenge on the host from the board's messages: nothing precomputed
        "CHALLENGES": " ".join(gold["challenges"]) if args.mode == "auto" else "",
        "POLL_MAX": args.poll_max,
    }
    hw_tcl = os.path.join(GEN_DIR, f"board_test_{tag}.tcl")
    with open(hw_tcl, "w", newline="\n") as fh:
        fh.write(render({**common, "DRY_RUN": 0, "EMU_MSGS": "", "EMU_FINALS": ""}))
    dry_tcl = os.path.join(GEN_DIR, f"board_test_{tag}_dryrun.tcl")
    fins = [w for pair in zip(gold["lane_final_f"], gold["lane_final_g"]) for w in pair]
    with open(dry_tcl, "w", newline="\n") as fh:
        fh.write(render({**common, "DRY_RUN": 1, "EMU_MSGS": " ".join(w for m in gold["msgs"] for w in m),
                         "EMU_FINALS": " ".join(fins)}))

    rec = {
        "record": "sumfold_board_test",
        "implementation": "rtl",
        "board": "ZCU102 (xczu9eg-ffvb1156-2-e), JTAG-to-AXI",
        "mode": args.mode, "lanes": L, "n": n, "device_rounds": R,
        "seed_f": f"{seed_f:016x}", "seed_g": f"{seed_g:016x}",
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "tools": {"vivado": "2022.2", "python": sys.version.split()[0]},
        "generated_tcl": {"hardware": fwd(os.path.relpath(hw_tcl, ROOT)), "dry_run": fwd(os.path.relpath(dry_tcl, ROOT))},
        "golden_seconds_python": round(t_gold, 3),
        "bitstream": fwd(os.path.relpath(os.path.abspath(args.bit), ROOT)),
        "bitstream_sha256": (hashlib.sha256(open(args.bit, "rb").read()).hexdigest()
                             if os.path.exists(args.bit) else None),
        "ltx": fwd(args.ltx) or None,
        "programmed": not args.no_program,
    }

    if not args.run:
        rec.update({"status": "NOT_RUN",
                    "evidence_class": "SIMULATION",
                    "evidence_note": "hardware NOT_RUN; the only evidence is the dry run of the host Tcl flow "
                                     "against a register emulator (checks the host script and the Tcl "
                                     "transcript, not the FPGA)",
                    "invocation": "python " + " ".join([fwd(os.path.relpath(__file__, ROOT))] + sys.argv[1:]),
                    "note": "hardware not executed (board programming not authorized for this task); dry run only"})
        if not args.no_vivado:
            log = os.path.join(GEN_DIR, f"board_test_{tag}_dryrun.log")
            rc, cmd, el = run_vivado(dry_tcl, log)
            text = open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else ""
            kv = parse_sfb(text)
            chk = check(kv, gold, args.mode, L)
            rec["dry_run"] = {"command": cmd, "exit_code": rc, "elapsed_s": round(el, 1),
                              "log": fwd(os.path.relpath(log, ROOT)),
                              "tcl_patchlevel": kv.get("TCL_PATCHLEVEL"), "tcllib_sha256": kv.get("SHA256_TCLLIB"),
                              "result": "PASS" if chk["match"] else "FAIL", **chk,
                              "what_it_checks": "control flow and register sequencing against an emulator that "
                                                "returns golden values; in STEP mode the challenges are computed "
                                                "by the Tcl SHA-256 transcript and compared with the golden model"}
        out = args.out or os.path.join(ROOT, "results", "raw", "fpga_sim", f"board_test_dryrun_{tag}.json")
    else:
        log = os.path.join(GEN_DIR, f"board_test_{tag}_{dt.datetime.now():%Y%m%d_%H%M%S}.log")
        rc, cmd, el = run_vivado(hw_tcl, log)
        text = open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else ""
        kv = parse_sfb(text)
        chk = check(kv, gold, args.mode, L)
        kclk = float(kv.get("KCLK_MHZ", "nan"))
        cyc = {k: int(v) for k, v in kv.items() if k.startswith("CYC_") and v.isdigit()}
        rec.update({
            "invocation": "python " + " ".join([fwd(os.path.relpath(__file__, ROOT))] + sys.argv[1:]),
            "status": "MEASURED_SUCCESS" if chk["match"] else "VERIFY_FAILED",
            "evidence_class": "MEASURED",
            "command": cmd, "exit_code": rc, "elapsed_s": round(el, 1), "log": fwd(os.path.relpath(log, ROOT)),
            "kclk_mhz_nominal": kclk, "cycles": cyc,
            "device_time_us_at_nominal_clock": {k: v / kclk for k, v in cyc.items()} if kclk == kclk else None,
            "host_times_us": {k: int(v) for k, v in kv.items() if k.endswith("_US") and v.isdigit()},
                        # STEP: per-round host times (POLL = START write to WAIT observed, HOST = WAIT observed to next START write: 3 message reads, Tcl SHA-256 transcript, challenge write)
            "step_round_us": ({"poll": [int(kv[f"STEP_POLL_US_{i}"]) for i in range(R) if f"STEP_POLL_US_{i}" in kv],
                               "service": [int(kv[f"STEP_HOST_US_{i}"]) for i in range(R) if f"STEP_HOST_US_{i}" in kv]}
                              if args.mode == "step" else None),
            "sysmon": {k[len("SYSMON_"):]: v for k, v in kv.items() if k.startswith("SYSMON_")} or None,
            **chk,
        })
        out = args.out or os.path.join(ROOT, "results", "raw", "fpga_board", f"board_test_{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="\n") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({k: rec[k] for k in ("status", "mode", "lanes", "n")}
                     | ({"dry_run": rec["dry_run"]["result"], "errors": rec["dry_run"]["errors"][:5]}
                        if "dry_run" in rec else {})
                     | ({"match": rec.get("match")} if args.run else {}), indent=1))
    print("wrote", out)
    print("hardware script:", hw_tcl)
    return 0

if __name__ == "__main__":
    sys.exit(main())
