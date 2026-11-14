r"""Run the whole EDGE-CERT figure / number / table pipeline from results/.

    python bench/analysis/make_all.py                # loc -> figures -> numbers -> tables
    python bench/analysis/make_all.py --latex-check  # + compile every table and chart with the
                                                     #   real preamble in a temporary directory and report
                                                     #   LaTeX errors / overfull boxes / undefined macros

Outputs: paper/figures/charts/*.pdf|svg + charts_report.json, paper/generated/numbers.tex, NUMBERS.md, generator.json, tables_report.json, make_all_report.json, paper/tables/gen_*.tex. Missing inputs never stop the run: they print "PENDING <name>: missing <file>"; re-run once they exist.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

STEPS = [("loc", "loc_count.py"), ("generator", "generator.py"), ("figures", "make_figures.py"),
         ("numbers", "make_numbers.py"), ("tables", "make_tables.py")]

def run_step(script: str) -> dict:
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / script)], cwd=str(C.REPO), capture_output=True, text=True)
    out = r.stdout + r.stderr
    lines = [ln for ln in out.splitlines() if "UserWarning" not in ln and "warnings.warn" not in ln]
    for ln in lines:
        print("   " + ln)
    return {"script": f"bench/analysis/{script}", "returncode": r.returncode, "seconds": round(time.time() - t0, 1),
            "pending": [ln for ln in lines if ln.startswith("PENDING ")],
            "checks": [ln for ln in lines if ln.startswith("[CHECK]")]}

def latex_check() -> dict:
    """Compile a two-column test document with the paper preamble, numbers.tex, every gen_*.tex table and
    every chart at its nominal width; report errors, overfull boxes and undefined control sequences."""
    charts = sorted((C.CHARTS).glob("fig_*.pdf"))
    tables = sorted((C.TABLES).glob("gen_*.tex"))
    tmp = Path(tempfile.mkdtemp(prefix="edgecert_latexcheck_"))
    for f in ("preamble.tex", "usenix.sty", "references.bib"):
        shutil.copy2(C.REPO / "paper" / f, tmp / f)
    (tmp / "generated").mkdir()
    shutil.copy2(C.GEN / "numbers.tex", tmp / "generated" / "numbers.tex")
    (tmp / "tables").mkdir()
    for t in tables:
        shutil.copy2(t, tmp / "tables" / t.name)
    (tmp / "charts").mkdir()
    for c in charts:
        shutil.copy2(c, tmp / "charts" / c.name)
    wide = {"fig_contract_lookups", "fig_memory_context", "fig_latency_cdf", "fig_kernel_throughput",
            "fig_breakeven", "fig_latency_breakdown", "fig_ablation"}
    body = []
    for c in charts:
        env = "figure*" if c.stem in wide else "figure"
        width = r"\textwidth" if c.stem in wide else r"\columnwidth"
        body.append(f"\\begin{{{env}}}[t]\\centering\\includegraphics[width={width}]{{charts/{c.name}}}"
                    f"\\caption{{{c.stem.replace('_', ' ')}}}\\end{{{env}}}")
    for t in tables:
        body.append(f"\\input{{tables/{t.name}}}")
    nums = re.findall(r"\\newcommand\{\\(Num[A-Za-z]+)\}", (C.GEN / "numbers.tex").read_text(encoding="utf-8"))
    body.append("\\section*{All numbers}" + " ".join(f"\\{n}{{}}" for n in nums))
    doc = ("\\documentclass[letterpaper,twocolumn,10pt]{article}\n\\input{preamble.tex}\n"
           "\\input{generated/numbers.tex}\n\\begin{document}\n\\lipsumfill\n" + "\n\n".join(body) +
           "\n\\end{document}\n").replace("\\lipsumfill\n", "Generated-artifact compile check.\n\n")
    (tmp / "check.tex").write_text(doc, encoding="utf-8")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "check.tex"], cwd=str(tmp),
                       capture_output=True, text=True, timeout=300)
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "check.tex"], cwd=str(tmp),
                       capture_output=True, text=True, timeout=300)
    log = (tmp / "check.log").read_text(encoding="latin-1") if (tmp / "check.log").exists() else ""
    over = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)[^\n]*\n?[^\n]*", log)
    over_big = [x for x in over if float(x) > 1.0]
    errs = [ln for ln in log.splitlines() if ln.startswith("!")]
    undef = [ln for ln in log.splitlines() if "Undefined control sequence" in ln]
    pages = None
    m = re.search(r"Output written on check\.pdf \((\d+) pages", log)
    if m:
        pages = int(m.group(1))
    res = {"dir": str(tmp), "returncode": r.returncode, "pages": pages, "errors": errs[:20],
           "undefined": undef[:20], "overfull_pt": over, "overfull_gt_1pt": over_big,
           "charts": [c.name for c in charts], "tables": [t.name for t in tables], "macros": len(nums)}
    print(f"   latex check: rc={r.returncode} pages={pages} errors={len(errs)} overfull>1pt={over_big} dir={tmp}")
    return res

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex-check", action="store_true")
    args = ap.parse_args()
    rep = {"record": "edgecert_make_all", "command": "python bench/analysis/make_all.py"
           + (" --latex-check" if args.latex_check else ""), "timestamp": C.now_iso(), "host": C.host_info(),
           "steps": {}}
    rc = 0
    for name, script in STEPS:
        print(f"== {name}: {script}")
        rep["steps"][name] = run_step(script)
        rc |= rep["steps"][name]["returncode"]
    if args.latex_check:
        print("== latex check")
        try:
            rep["latex_check"] = latex_check()
        except Exception as e:  # pdflatex missing or timeout: report, do not fail the pipeline
            rep["latex_check"] = {"error": repr(e)}
            print("   latex check failed: " + repr(e))
    pend = sorted({p for s in rep["steps"].values() for p in s["pending"]})
    rep["pending"] = pend
    C.write_json(C.GEN / "make_all_report.json", rep)
    print(f"== done: rc={rc}; {len(pend)} PENDING lines; report paper/generated/make_all_report.json")
    return rc

if __name__ == "__main__":
    sys.exit(main())
