"""Shared helpers for the SumFold kernel runners and tests (CPU C / CUDA).

* build_cpu(): compile kernels/cpu with MinGW-w64 gcc (default) or MSVC.
* run_cpu(): run sumfold_cpu.exe and parse its JSON record.
* golden_record(): the golden-model transcript (edgecert.sumfold.prove) in the same
  hex layout as the kernels ({c1,c0} 128-bit words).
* compare_records(): field-by-field bit-exactness check.
* gen_vector_fast(): numpy xorshift64 generator (bit-identical to sumfold.gen_vector)
  using GF(2) jump-ahead, for large N.
* host_info(), contended_now(), stats helpers (median, p95, bootstrap CI).

Standard library + numpy only.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from edgecert import sumfold as _sf  # noqa: E402  (golden model, read-only)
from edgecert.field import P, ehex  # noqa: E402

CPU_DIR = ROOT / "kernels" / "cpu"
CPU_BUILD = CPU_DIR / "build"
CPU_SOURCES = ["main.c", "sumfold.c", "sha256.c"]
CPU_HEADERS = ["gl_field.h", "sumfold.h", "sha256.h"]
MINGW_BIN = Path(r"<vivado-2022.2>\tps\mingw\6.2.0\win64.o\nt\bin")
MSVC_VCVARS = Path(r"<mscv-vcvars>"
                   r"\VC\Auxiliary\Build\vcvars64.bat")

GCC_FLAGS = ["-O3", "-march=native", "-funroll-loops", "-fopenmp", "-static", "-std=gnu11", "-Wall"]
MSVC_FLAGS = ["/O2", "/openmp", "/nologo", "/W3", "/D_CRT_SECURE_NO_WARNINGS"]

DEFAULT_SEED_F = 0x123456789ABCDEF1
DEFAULT_SEED_G = 0x0FEDCBA987654321


def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------- build
def _gcc_env() -> dict:
    env = dict(os.environ)
    env["PATH"] = str(MINGW_BIN) + os.pathsep + env.get("PATH", "")
    return env


def cpu_exe(compiler: str = "gcc") -> Path:
    return CPU_BUILD / ("sumfold_cpu.exe" if compiler == "gcc" else "sumfold_cpu_msvc.exe")


def build_cpu(force: bool = False, compiler: str = "gcc", extra_flags: Sequence[str] = ()) -> Dict:
    """Compile kernels/cpu; returns {'exe', 'compiler', 'flags', 'command', 'rebuilt'}."""
    CPU_BUILD.mkdir(parents=True, exist_ok=True)
    exe = cpu_exe(compiler)
    srcs = [CPU_DIR / s for s in CPU_SOURCES]
    deps = srcs + [CPU_DIR / h for h in CPU_HEADERS]
    stale = force or not exe.exists() or any(d.stat().st_mtime > exe.stat().st_mtime for d in deps)
    if compiler == "gcc":
        flags = GCC_FLAGS + list(extra_flags)
        flag_str = " ".join(flags)
        cmd = [str(MINGW_BIN / "gcc.exe")] + flags + [f'-DSF_BUILD_FLAGS="{flag_str}"', "-o", str(exe)] + \
              [str(s) for s in srcs]
        if stale:
            r = subprocess.run(cmd, cwd=str(CPU_DIR), env=_gcc_env(), capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError("gcc build failed:\n" + r.stdout + r.stderr)
        return {"exe": str(exe), "compiler": "gcc 6.2.0 (MinGW-w64 x86_64-win32-sjlj-rev1, Vivado tps)",
                "flags": flag_str, "command": subprocess.list2cmdline(cmd), "rebuilt": stale}
    if compiler == "msvc":
        flags = MSVC_FLAGS + list(extra_flags)
        flag_str = " ".join(flags)
        inner = " ".join(flags) + f' /DSF_BUILD_FLAGS="\\"{flag_str}\\"" /Fe:"{exe}" /Fo:"{CPU_BUILD}\\\\" ' + \
            " ".join(f'"{s}"' for s in srcs)
        bat = CPU_BUILD / "_build_msvc.bat"
        bat.write_text(f'@echo off\ncall "{MSVC_VCVARS}" >nul\ncl {inner}\n', encoding="ascii")
        if stale:
            r = subprocess.run(["cmd", "/c", str(bat)], cwd=str(CPU_DIR), capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError("msvc build failed:\n" + r.stdout + r.stderr)
        return {"exe": str(exe), "compiler": "msvc (VS BuildTools 18)", "flags": flag_str,
                "command": f"cl {inner}", "rebuilt": stale}
    raise ValueError(compiler)


# ----------------------------------------------------------------------- run
def cpu_command(exe: str, logn: int, threads: int, variant: str, seedf: int, seedg: int,
                reps: int, warmup: int, out: Optional[str] = None, lanes: Optional[int] = None,
                sched: Optional[str] = None, affinity: Optional[int] = None) -> List[str]:
    cmd = [exe, "--logn", str(logn), "--threads", str(threads), "--variant", variant,
           "--seedf", f"{seedf:016x}", "--seedg", f"{seedg:016x}", "--reps", str(reps),
           "--warmup", str(warmup)]
    if lanes is not None:
        cmd += ["--lanes", str(lanes)]
    if sched is not None:
        cmd += ["--sched", sched]
    if affinity:
        cmd += ["--affinity", f"{affinity:x}"]
    if out is not None:
        cmd += ["--out", out]
    return cmd


def run_cpu(exe: str, logn: int, threads: int, variant: str, seedf: int = DEFAULT_SEED_F,
            seedg: int = DEFAULT_SEED_G, reps: int = 1, warmup: int = 0, lanes: Optional[int] = None,
            sched: Optional[str] = None, affinity: Optional[int] = None, timeout: float = 3600.0) -> Dict:
    cmd = cpu_command(exe, logn, threads, variant, seedf, seedg, reps, warmup, None, lanes, sched, affinity)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{subprocess.list2cmdline(cmd)} failed rc={r.returncode}: {r.stderr}")
    d = json.loads(r.stdout)
    d["_command"] = subprocess.list2cmdline(cmd)
    return d


# -------------------------------------------------------------------- golden
_GOLDEN_CACHE: Dict = {}


def golden_record(seedf: int, seedg: int, n: int) -> Dict:
    key = (seedf, seedg, n)
    if key not in _GOLDEN_CACHE:
        f = _sf.gen_vector(seedf, 1 << n)
        g = _sf.gen_vector(seedg, 1 << n)
        pf = _sf.prove(f, g)
        _GOLDEN_CACHE[key] = {
            "claim": ehex(pf.claim),
            "msgs": [[ehex(s) for s in m] for m in pf.msgs],
            "challenges": [ehex(r) for r in pf.challenges],
            "final_f": ehex(pf.final_f),
            "final_g": ehex(pf.final_g),
        }
    return _GOLDEN_CACHE[key]


TRANSCRIPT_FIELDS = ("claim", "msgs", "challenges", "final_f", "final_g")


def compare_records(a: Dict, b: Dict) -> List[str]:
    """Return the list of transcript fields that differ (empty list = bit-exact)."""
    return [k for k in TRANSCRIPT_FIELDS if a.get(k) != b.get(k)]


def verify_record(rec: Dict) -> bool:
    """Run the golden verifier (transcript chain + final product) on a kernel record."""
    pe = _sf.parse_ehex
    pf = _sf.SumFoldProof(n=rec["n"], claim=pe(rec["claim"]),
                          msgs=[tuple(pe(x) for x in m) for m in rec["msgs"]],
                          challenges=[pe(x) for x in rec["challenges"]],
                          final_f=pe(rec["final_f"]), final_g=pe(rec["final_g"]))
    return _sf.verify(pf)


def transcript_digest(rec: Dict) -> str:
    import hashlib
    blob = json.dumps({k: rec[k] for k in TRANSCRIPT_FIELDS}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


# ------------------------------------------------ fast xorshift64 generator
def _xorshift_matrix() -> List[int]:
    """Columns of the GF(2) 64x64 matrix T with xorshift64(x) = T x (column j = T e_j)."""
    return [_sf.xorshift64(1 << j) for j in range(64)]


def _mat_apply(cols: List[int], x: int) -> int:
    y = 0
    j = 0
    while x:
        if x & 1:
            y ^= cols[j]
        x >>= 1
        j += 1
    return y


def _mat_mul(a: List[int], b: List[int]) -> List[int]:
    return [_mat_apply(a, bc) for bc in b]  # (A B) e_j = A (B e_j)


def _mat_pow(cols: List[int], e: int) -> List[int]:
    result = [1 << j for j in range(64)]
    base = cols
    while e:
        if e & 1:
            result = _mat_mul(base, result)
        base = _mat_mul(base, base)
        e >>= 1
    return result


def gen_vector_fast(seed: int, n_elems: int, streams: int = 0):
    """numpy uint64 array, bit-identical to [c0 for (c0, _) in sumfold.gen_vector(seed, n)]."""
    import numpy as np
    assert seed != 0
    if streams <= 0:
        streams = 1
        while streams * streams < n_elems:
            streams <<= 1
    streams = min(streams, n_elems)
    chunk = -(-n_elems // streams)
    T = _xorshift_matrix()
    J = _mat_pow(T, chunk)
    starts = [seed]
    for _ in range(streams - 1):
        starts.append(_mat_apply(J, starts[-1]))
    x = np.array(starts, dtype=np.uint64)
    out = np.empty((chunk, streams), dtype=np.uint64)
    s13, s7, s17 = np.uint64(13), np.uint64(7), np.uint64(17)
    for k in range(chunk):
        x ^= x << s13
        x ^= x >> s7
        x ^= x << s17
        out[k] = x
    v = out.T.reshape(-1)[:n_elems].copy()
    p = np.uint64(P)
    v[v >= p] -= p
    return v


# ------------------------------------------------------- CPU topology / pin
def cpu_topology() -> List[Dict]:
    """Physical cores of processor group 0 (Windows): [{'efficiency_class', 'mask', 'logical'}].

    Uses GetLogicalProcessorInformationEx(RelationProcessorCore).  On hybrid Intel parts
    P-cores report a higher EfficiencyClass than E-cores.  Empty list if unavailable.
    """
    if os.name != "nt":
        return []
    import ctypes
    import struct
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ln = ctypes.c_uint32(0)
    k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(ln))
    buf = ctypes.create_string_buffer(ln.value)
    if not k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(ln)):
        return []
    raw = buf.raw[:ln.value]
    cores, off = [], 0
    while off < len(raw):
        rel, size = struct.unpack_from("<II", raw, off)
        if rel == 0:  # RelationProcessorCore
            eff = raw[off + 9]
            gcount = struct.unpack_from("<H", raw, off + 30)[0]
            for gi in range(gcount):
                mask, group = struct.unpack_from("<QH", raw, off + 32 + 16 * gi)
                if group == 0:
                    cores.append({"efficiency_class": eff, "mask": mask,
                                  "logical": [b for b in range(64) if mask >> b & 1]})
        off += size
    return cores


def affinity_order(cores: List[Dict]) -> List[int]:
    """Logical CPUs ordered: first thread of each fastest core, then slower cores, then
    SMT siblings (so t threads land on t distinct physical cores whenever t <= #cores)."""
    # fastest class first; inside a class highest logical index first, so a 1-thread run
    # avoids logical CPU 0 (which services most interrupts/DPCs on Windows)
    by_cls = sorted(cores, key=lambda c: (-c["efficiency_class"], -c["logical"][0]))
    first = [c["logical"][0] for c in by_cls]
    rest = [lp for c in by_cls for lp in c["logical"][1:]]
    return first + rest


def affinity_mask(threads: int, policy: str = "pfirst") -> Optional[int]:
    """Process affinity mask for 'threads' threads under a pinning policy (None = OS default)."""
    if policy == "none":
        return None
    order = affinity_order(cpu_topology())
    if not order:
        return None
    m = 0
    for lp in order[:max(1, min(threads, len(order)))]:
        m |= 1 << lp
    return m


# ---------------------------------------------------------------- host info
def _ps(cmd: str) -> str:
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True,
                           text=True, timeout=60)
        return r.stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"error: {e}"


def host_info() -> Dict:
    info = {"platform": platform.platform(), "python": sys.version.split()[0],
            "machine": platform.machine(), "logical_cpus": os.cpu_count()}
    if os.name == "nt":
        cpu = _ps("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,"
                  "NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json -Compress")
        try:
            info["cpu"] = json.loads(cpu)
        except Exception:
            info["cpu"] = cpu
        mem = _ps("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")
        try:
            info["ram_GiB"] = round(int(mem) / 2**30, 1)
        except Exception:
            info["ram_GiB"] = mem
        os_ = _ps("Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber"
                  " | ConvertTo-Json -Compress")
        try:
            info["os"] = json.loads(os_)
        except Exception:
            info["os"] = os_
        info["power_plan"] = _ps("powercfg /getactivescheme")
    return info


CONTENTION_PATTERN = r"vivado|xsim|xelab|xvlog|xvhdl|loader|hw_server|cs_server|parallel_synth"


def contended_now(window_s: float = 1.0) -> Dict:
    """Snapshot of Vivado/xsim processes (Get-Process); contended = any found.

    Also measures how busy they are: CPU seconds consumed by the matching processes during
    window_s, as a percentage of all logical CPUs ('active_cpu_pct'); an idle Vivado shell
    shows ~0 %, a synthesis/implementation run shows tens of percent."""
    if os.name != "nt":
        return {"contended": None, "processes": []}
    cmd = (f"$a = Get-Process | Where-Object {{ $_.ProcessName -match '{CONTENTION_PATTERN}' }}; "
           f"$c0 = ($a | Measure-Object -Property CPU -Sum).Sum; Start-Sleep -Milliseconds {int(window_s * 1000)}; "
           f"$b = Get-Process | Where-Object {{ $_.ProcessName -match '{CONTENTION_PATTERN}' }}; "
           "$c1 = ($b | Measure-Object -Property CPU -Sum).Sum; "
           "$ld = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average; "
           "@{procs = @($b | Select-Object ProcessName,Id); c0 = $c0; c1 = $c1; load = $ld} | "
           "ConvertTo-Json -Compress -Depth 3")
    out = _ps(cmd)
    procs, act, load = [], None, None
    if out and not out.startswith("error"):
        try:
            j = json.loads(out)
            procs = j.get("procs") or []
            if isinstance(procs, dict):
                procs = [procs]
            if j.get("c0") is not None and j.get("c1") is not None:
                act = max(0.0, (float(j["c1"]) - float(j["c0"])) / (window_s * (os.cpu_count() or 1)) * 100)
            if j.get("load") is not None:
                load = float(j["load"])
        except Exception:
            procs = [{"raw": out[:300]}]
    return {"contended": bool(procs), "processes": procs, "active_cpu_pct": act,
            "machine_load_pct_while_idle": load, "host_busy": (load is not None and load > 15.0),
            "t": now_utc()}


def cpu_load_percent() -> Optional[float]:
    out = _ps("(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average")
    try:
        return float(out)
    except Exception:
        return None


# ------------------------------------------------------------------- stats
def percentile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy 'linear' method), q in [0, 100]."""
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def bootstrap_ci_median(xs: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                        seed: int = 12345) -> List[float]:
    rng = random.Random(seed)
    xs = list(xs)
    meds = sorted(statistics.median(rng.choices(xs, k=len(xs))) for _ in range(n_boot))
    return [meds[int(n_boot * alpha / 2)], meds[min(n_boot - 1, int(n_boot * (1 - alpha / 2)))]]


def summarize(xs: Sequence[float]) -> Dict:
    return {"median": statistics.median(xs), "p95": percentile(xs, 95), "min": min(xs), "max": max(xs),
            "mean": statistics.fmean(xs), "ci95_median_bootstrap": bootstrap_ci_median(xs), "n": len(xs)}


def tool_available(name: str) -> Optional[str]:
    return shutil.which(name)
