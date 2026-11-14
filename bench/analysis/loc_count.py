"""Line-of-code count (non-blank, non-comment-only lines) per prototype component -> paper/generated/loc.json.

Run from the repository root:  python bench/analysis/loc_count.py

Moved into the repository by the fixer (2026-09-29) from the integrator's scratch script so that the component table of the paper (tab:loc, paper/tables/gen_loc.tex written by make_tables.py) has a reproducible source (review finding 37). Counting rules are unchanged: same rules as the writer-design snapshot (docs/notes/writer_design.md, section 4), with the component globs extended by the files added after that snapshot (netlist and replay simulation, board scripts, chart/table/number generators). Read-only on the sources; the only output is paper/generated/loc.json. Evidence class: MEASURED (static count).
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "generated" / "loc.json"
COMMENT = {".py": ("#",), ".tcl": ("#",), ".tex": ("%",), ".vh": ("//",)}
BLOCK = (".c", ".h", ".cu", ".cpp", ".v", ".sv")      # // and /* */ comments
COMPONENTS = [
    ("Golden model", "Python", ["src/edgecert/field.py", "src/edgecert/sumfold.py"]),
    ("Contract, session, verifier (C1)", "Python",
     ["src/edgecert/runtime/manifest.py", "src/edgecert/runtime/session.py",
      "src/edgecert/crypto/*.py", "src/edgecert/verifier/*.py", "src/edgecert/audit/*.py"]),
    ("Prover adapters, compact model", "Python", ["src/edgecert/prover_adapters/*.py"]),
    ("Integer reference, TileContract (C2)", "Python",
     ["src/edgecert/reference/*.py", "src/edgecert/compiler/*.py"]),
    ("CertSched and simulator (C3)", "Python",
     ["src/edgecert/runtime/dag.py", "src/edgecert/runtime/scheduler.py",
      "src/edgecert/runtime/simulate.py"]),
    ("CPU kernels", "C", ["kernels/cpu/*.c", "kernels/cpu/*.h"]),
    ("GPU kernels, test shim", "CUDA, C++",
     ["kernels/cuda/*.cu", "kernels/cuda/shim/*.cpp", "kernels/cuda/shim/*.h"]),
    ("SumFold RTL (C4)", "Verilog", ["fpga/rtl/*.v", "fpga/rtl/*.vh"]),
    ("RTL and netlist testbenches", "SystemVerilog",
     ["fpga/tb/*.sv", "fpga/tb/*.v", "fpga/sim/netlist/*.sv"]),
    ("FPGA host, build and replay scripts", "Python, Tcl",
     ["fpga/host/*.py", "fpga/scripts/*.py", "fpga/scripts/*.tcl", "fpga/sim/*.py",
      "fpga/sim/ooc/*.py", "fpga/sim/ooc/*.tcl", "fpga/sim/netlist/*.py", "fpga/sim/netlist/*.tcl",
      "fpga/sim/replay/*.py"]),
    ("Benchmark and attack runners", "Python", ["bench/runners/*.py", "bench/attacks/*.py"]),
    ("Chart, table and number generators", "Python", ["bench/analysis/*.py"]),
    ("Unit tests", "Python", ["tests/*.py"]),
    ("Schematic figures", "TikZ", ["paper/figures/schematic/fig_*.tex"]),
]

def count(p):
    n, blk = 0, False
    for s in (l.strip() for l in p.read_text(encoding="utf-8", errors="replace").splitlines()):
        if not s:
            continue
        if p.suffix in BLOCK:
            if blk:
                blk = "*/" not in s
                continue
            if s.startswith("/*"):
                blk = "*/" not in s
                continue
            if s.startswith("//"):
                continue
        elif any(s.startswith(c) for c in COMMENT.get(p.suffix, ())):
            continue
        n += 1
    return n

def main() -> int:
    rows = []
    for name, lang, globs in COMPONENTS:
        files = sorted({f for g in globs for f in ROOT.glob(g) if f.is_file()})
        rows.append({"component": name, "language": lang, "files": len(files),
                     "lines": sum(count(f) for f in files)})
    out = {"record": "edgecert_loc_count", "status": "MEASURED_SUCCESS", "evidence_class": "MEASURED",
           "what": "non-blank, non-comment-only source lines per component (static count)",
           "command": "python bench/analysis/loc_count.py",
           "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "contention_flags": {"recorded": False, "note": "not applicable (static count)"},
           "rows": rows,
           "total_files": sum(r["files"] for r in rows),
           "total_lines": sum(r["lines"] for r in rows)}
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}: {out['total_files']} files, {out['total_lines']} lines")
    return 0

if __name__ == "__main__":
    sys.exit(main())
