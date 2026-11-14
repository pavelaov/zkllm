#!/usr/bin/env python3
"""Compare the SFB_* values of an xsim AXI replay (results/raw/fpga_sim/replay_<tag>.json, key "sfb") with
the SFB_* lines of a board log (fpga/host/generated/board_test_*.log). Writes a SIMULATION-vs-MEASURED
comparison record.

  python fpga/sim/replay/compare_replay_board.py --replay results/raw/fpga_sim/replay_L1_n10_mulnet.json \
      --board-log fpga/host/generated/board_test_auto_L1_n10_20260927_100314.log \
      --out results/raw/fpga_sim/replay_L1_n10_mulnet_vs_board.json
"""
import argparse
import datetime as dt
import json
import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--board-log", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = json.load(open(os.path.join(ROOT, a.replay)))["sfb"]
    log = open(os.path.join(ROOT, a.board_log), errors="replace").read()
    brd = dict(re.findall(r"^SFB_(\w+)=(.*)$", log, flags=re.M))
    pat = r"(MSG|FIN|CYC_PASS|CYC_TOTAL|CYC_GEN|STATUS|ROUND)"
    keys = [k for k in brd if re.match(pat, k)]
    rows, cats = [], {}
    for k in keys:
        got = rep.get(k)
        ok = got is not None and str(got).strip() == brd[k].strip()
        c = re.match(pat, k).group(1)
        cats.setdefault(c, {"compared": 0, "equal": 0})
        cats[c]["compared"] += 1
        cats[c]["equal"] += ok
        rows.append({"key": k, "board": brd[k].strip(), "replay": got, "equal": ok})
    rec = {"record": "replay_vs_board", "status": "COMPLETED",
           "evidence_class": "SIMULATION (replay) compared with MEASURED (board log)",
           "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "command": "python " + " ".join(["fpga/sim/replay/compare_replay_board.py"] + sys.argv[1:]),
           "contention_flags": None, "replay": a.replay, "board_log": a.board_log,
           "compared": len(rows), "equal": sum(r["equal"] for r in rows), "by_category": cats,
           "all_equal": all(r["equal"] for r in rows) and bool(rows), "rows": rows}
    with open(os.path.join(ROOT, a.out), "w", newline="\n") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({k: rec[k] for k in ("compared", "equal", "all_equal", "by_category")}))

if __name__ == "__main__":
    main()
