#!/usr/bin/env python3
"""Compare an xsim replay of a board_probe.py trace (replay.py --probe-trace) with the MEASURED probe record of the same trace: every 32-bit read in order, and the assembled 128-bit words by key. usage: python fpga/sim/replay/compare_probe.py <trace.keys.json> <replay work dir> <board probe json> [out.json]"""
import json
import os
import sys

keys = json.load(open(sys.argv[1]))["keys"]
vals = [v.strip() for v in open(os.path.join(sys.argv[2], "values.txt")) if v.strip()]
board = json.load(open(sys.argv[3]))
sim = dict(zip(keys, vals))

def w128(d, key):
    try:
        w = [int(d[f"{key}.w{k}"], 16) for k in range(4)]
    except KeyError:
        return None
    return f"{w[3]:08x}{w[2]:08x}{w[1]:08x}{w[0]:08x}"

rows = []
for r in board["rows"]:
    s = w128(sim, r["key"])
    rows.append({"key": r["key"], "board": r["got"], "sim": s, "golden": r["exp"], "sim_eq_board": s == r["got"]})
same = sum(x["sim_eq_board"] for x in rows)
out = {"record": "sumfold_replay_vs_board_probe", "evidence_class": "SIMULATION vs MEASURED",
       "trace_keys": sys.argv[1], "replay_work": sys.argv[2], "board_probe": sys.argv[3],
       "reads_sim": len(vals), "reads_expected": len(keys), "words_compared": len(rows),
       "words_sim_equal_board": same, "words_board_equal_golden": sum(x["board"] == x["golden"] for x in rows),
       "rows": rows}
print(json.dumps({k: out[k] for k in ("reads_sim", "reads_expected", "words_compared", "words_sim_equal_board",
                                         "words_board_equal_golden")}))
for x in rows:
    if not x["sim_eq_board"]:
        print("DIFF", x["key"], "board", x["board"], "sim", x["sim"])
if len(sys.argv) > 4:
    json.dump(out, open(sys.argv[4], "w", newline="\n"), indent=1)
