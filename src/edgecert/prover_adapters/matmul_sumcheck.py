"""TRANSPARENT_REFERENCE matrix-multiplication sumcheck prover and verifier (NOT zero knowledge, NOT succinct, no polynomial commitment).

Relation per matmul  C = A W  (A: T x K, W: K x N, C: T x N, small signed/unsigned integers lifted to Goldilocks, |C| < p/2 so the field identity is the integer identity):

  1. instance digest  D = SHA-256("EDGE-CERT/gpt2prove/instance/v1" || layer id || shapes ||
     digest(A) || digest(W) || digest(C)),  digest(X) = SHA-256 of the C-order bytes of X
     with a dtype/shape header.  digest(W) is the model commitment stand-in (computed once).
  2. Fiat-Shamir (edgecert.crypto.transcript.ProofTranscript(D)): absorb the layer id, squeeze
     r1 in Ext^t (t = ceil log2 T) and r2 in Ext^nb (nb = ceil log2 N).
  3. claim c = C~(r1, r2).  Prover: a_k = A~(r1, k), w_k = W~(k, r2) for all k < K (MLE partial
     evaluation by repeated folds with fixed challenges, the first fold base x ext), then the
     SumFold product sumcheck over k (K zero-padded to 2^kappa) proving sum_k a_k w_k = c.  The
     sumcheck transcript is edgecert.crypto.transcript.BoundSumcheckTranscript(kappa, c,
     context = state of the proof transcript after r2, label = "<layer id>/sumcheck").
  4. Verifier (independent code path): recomputes D, r1, r2, C~(r1, r2) from the claimed C,
     checks every round (s0 + s1 = running claim, golden interp_quadratic), the final product,
     and the final evaluations a(r3) = A~(r1, r3), w(r3) = W~(r3, r2) by recomputing them from
     A and W (eq-table contraction, not folds).  A real system replaces step 4's recomputation
     by polynomial-commitment openings; here the verifier reads A and W (non-succinct).

Proof = claim, kappa round messages (3 Ext each), a(r3), w(r3):  16 + 48 kappa + 32 bytes.

Backends (all produce bit-identical proofs; tests/test_matmul_sumcheck*.py):
  PythonBackend  pure Python on the golden model (edgecert.sumfold.fold / round_message)
  CpuBackend     kernels/cpu/mle.c DLL (gcc 6.2, OpenMP) through ctypes
  GpuBackend     kernels/cuda/sumfold.cu + mle.cu through CuPy RawModule (NVRTC); batched
                 instances (attention heads) share one launch sequence.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field as dfield
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..field import P, Ext, ZERO, eadd, emul, enc, ehex, esub
from ..sumfold import fold as golden_fold, interp_quadratic, round_message
from ..crypto.transcript import BoundSumcheckTranscript, ProofTranscript

LABEL = "TRANSPARENT_REFERENCE"
PROOF_FORMAT = "edgecert-matmul-sumcheck/v1"
INSTANCE_DST = b"EDGE-CERT/gpt2prove/instance/v1"
MAT_DST = b"EDGE-CERT/gpt2prove/mat/v1"

ROOT = Path(__file__).resolve().parents[3]
CPU_DIR = ROOT / "kernels" / "cpu"
CUDA_DIR = ROOT / "kernels" / "cuda"
DLL_PATH = CPU_DIR / "build" / "edgecert_mle.dll"
MINGW_BIN = Path(r"<vivado-2022.2>\tps\mingw\6.2.0\win64.o\nt\bin")
DLL_FLAGS = ["-O3", "-march=native", "-funroll-loops", "-fopenmp", "-std=gnu11", "-Wall", "-shared", "-static"]
DLL_SOURCES = ["mle.c", "sumfold.c", "sha256.c"]
DLL_HEADERS = ["mle.h", "gl_field.h", "sumfold.h", "sha256.h"]

# element type codes (kernels/cpu/mle.h)
MLE_I8, MLE_U8, MLE_I16, MLE_U16, MLE_I32, MLE_I64, MLE_EXT = range(7)
_DT_CODE = {np.dtype(np.int8): MLE_I8, np.dtype(np.uint8): MLE_U8, np.dtype(np.int16): MLE_I16,
            np.dtype(np.uint16): MLE_U16, np.dtype(np.int32): MLE_I32, np.dtype(np.int64): MLE_I64}
MAX_VARS = 40

def log2_ceil(n: int) -> int:
    return 0 if n <= 1 else (n - 1).bit_length()

def dtype_code(a: np.ndarray) -> int:
    try:
        return _DT_CODE[a.dtype]
    except KeyError:
        raise TypeError(f"unsupported matrix dtype {a.dtype}") from None

def elem_strides(a: np.ndarray) -> Tuple[int, int]:
    s0, s1 = a.strides
    it = a.itemsize
    if s0 % it or s1 % it:
        raise ValueError("strides must be multiples of the item size")
    return s0 // it, s1 // it

# ============================================================ instance and transcript
def mat_digest(x: np.ndarray) -> bytes:
    """SHA-256 over a dtype/shape header and the C-order bytes of the logical matrix."""
    assert x.ndim == 2
    hdr = MAT_DST + x.dtype.str.encode() + int(x.shape[0]).to_bytes(8, "little") + \
        int(x.shape[1]).to_bytes(8, "little")
    h = hashlib.sha256(hdr)
    h.update(np.ascontiguousarray(x).data)
    return h.digest()

def instance_digest(layer_id: str, T: int, K: int, N: int, dA: bytes, dW: bytes, dC: bytes) -> bytes:
    lb = layer_id.encode()
    return hashlib.sha256(INSTANCE_DST + len(lb).to_bytes(4, "little") + lb + T.to_bytes(8, "little")
                          + K.to_bytes(8, "little") + N.to_bytes(8, "little") + dA + dW + dC).digest()

@dataclass
class MatmulInstance:
    """Public instance of one matmul C = A W (logical views; strided numpy views are fine)."""
    layer_id: str
    A: np.ndarray              # T x K
    W: np.ndarray              # K x N
    C: np.ndarray              # T x N (claimed output)
    w_digest: Optional[bytes] = None   # cached digest(W) (model commitment stand-in)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return int(self.A.shape[0]), int(self.A.shape[1]), int(self.W.shape[1])

    def check_shapes(self):
        T, K, N = self.shape
        if self.W.shape[0] != K or self.C.shape != (T, N):
            raise ValueError(f"{self.layer_id}: shape mismatch A{self.A.shape} W{self.W.shape} C{self.C.shape}")
        if K < 2:
            raise ValueError("K >= 2 required")

@dataclass
class Binding:
    digest: bytes
    r1: List[Ext]
    r2: List[Ext]
    ctx: bytes
    label: bytes
    t: int
    nb: int
    kappa: int

def bind(inst: MatmulInstance) -> Binding:
    """Instance digest and Fiat-Shamir challenges r1, r2 (shared protocol definition)."""
    T, K, N = inst.shape
    dW = inst.w_digest if inst.w_digest is not None else mat_digest(inst.W)
    D = instance_digest(inst.layer_id, T, K, N, mat_digest(inst.A), dW, mat_digest(inst.C))
    t, nb, kappa = log2_ceil(T), log2_ceil(N), max(1, log2_ceil(K))
    pt = ProofTranscript(D)
    pt.absorb("layer_id", inst.layer_id.encode())
    r1 = pt.challenge_vec("r1", t)
    r2 = pt.challenge_vec("r2", nb)
    return Binding(D, r1, r2, pt.state, (inst.layer_id + "/sumcheck").encode(), t, nb, kappa)

# ================================================================ proof object
@dataclass
class MatmulProof:
    layer_id: str
    T: int
    K: int
    N: int
    kappa: int
    claim: Ext
    msgs: List[Tuple[Ext, Ext, Ext]]
    final_a: Ext
    final_w: Ext
    challenges: List[Ext] = dfield(default_factory=list)   # prover-reported r3 (verifier re-derives)
    backend: str = ""
    label: str = LABEL

    def to_bytes(self) -> bytes:
        return enc(self.claim) + b"".join(enc(s) for m in self.msgs for s in m) + enc(self.final_a) + \
            enc(self.final_w)

    @property
    def n_bytes(self) -> int:
        return 16 + 48 * self.kappa + 32

    def digest(self) -> str:
        return hashlib.sha256(self.layer_id.encode() + b"|" + self.to_bytes()).hexdigest()

    def to_json(self) -> dict:
        return {"format": PROOF_FORMAT, "label": self.label, "layer_id": self.layer_id,
                "shape": [self.T, self.K, self.N], "kappa": self.kappa, "claim": ehex(self.claim),
                "msgs": [[ehex(s) for s in m] for m in self.msgs], "final_a": ehex(self.final_a),
                "final_w": ehex(self.final_w), "bytes": self.n_bytes}

def _ext_arr(xs: Sequence[Ext]) -> np.ndarray:
    a = np.zeros((max(len(xs), 1), 2), dtype=np.uint64)
    for i, (c0, c1) in enumerate(xs):
        a[i, 0] = c0
        a[i, 1] = c1
    return a

def _ext_list(a: np.ndarray, n: Optional[int] = None) -> List[Ext]:
    n = a.shape[0] if n is None else n
    return [(int(a[i, 0]), int(a[i, 1])) for i in range(n)]

# ================================================================ pure-Python backend
def py_fold_axis(M: np.ndarray, r: Sequence[Ext], nr: int) -> List[Ext]:
    """out[k] = M~(r, k) for the logical (nfold x nkeep) matrix M, by nr golden folds per column."""
    nfold, nkeep = M.shape
    assert nfold <= (1 << nr)
    rows = M.tolist()
    out = []
    for k in range(nkeep):
        col = [(int(rows[i][k]) % P, 0) for i in range(nfold)] + [ZERO] * ((1 << nr) - nfold)
        for level in range(nr):
            col = golden_fold(col, r[level])
        out.append(col[0])
    return out

def py_sumcheck(f: List[Ext], g: List[Ext], ctx: Optional[bytes], label: Optional[bytes]):
    """Golden product sumcheck; bound transcript when ctx is given (BoundSumcheckTranscript)."""
    from ..sumfold import Transcript, inner
    n = len(f).bit_length() - 1
    claim = inner(f, g)
    tr = BoundSumcheckTranscript(n, claim, ctx, label.decode()) if ctx is not None else Transcript(n, claim)
    msgs, chals = [], []
    cf, cg = list(f), list(g)
    for _ in range(n):
        m = round_message(cf, cg)
        r = tr.round(*m)
        msgs.append(m)
        chals.append(r)
        cf, cg = golden_fold(cf, r), golden_fold(cg, r)
    return claim, msgs, chals, cf[0], cg[0]

class PythonBackend:
    name = "python"

    def prove(self, inst: MatmulInstance, b: Optional[Binding] = None, threads: int = 1,
              want_vectors: bool = False):
        inst.check_shapes()
        T, K, N = inst.shape
        t0 = time.perf_counter()
        b = b or bind(inst)
        t1 = time.perf_counter()
        a = py_fold_axis(inst.A, b.r1, b.t)
        t2 = time.perf_counter()
        w = py_fold_axis(inst.W.T, b.r2, b.nb)
        t3 = time.perf_counter()
        pad = [ZERO] * ((1 << b.kappa) - K)
        claim, msgs, chals, fa, fw = py_sumcheck(a + pad, w + pad, b.ctx, b.label)
        t4 = time.perf_counter()
        pf = MatmulProof(inst.layer_id, T, K, N, b.kappa, claim, msgs, fa, fw, chals, backend=self.name)
        tm = {"bind_s": t1 - t0, "fold_a_s": t2 - t1, "fold_w_s": t3 - t2, "sumcheck_s": t4 - t3,
              "core_s": t4 - t1, "wall_s": t4 - t0}
        if want_vectors:
            tm["a"], tm["w"] = a, w
        return pf, tm

# ================================================================ C library
class MmTimes(ctypes.Structure):
    _fields_ = [("fold_a_s", ctypes.c_double), ("fold_w_s", ctypes.c_double),
                ("sumcheck_s", ctypes.c_double), ("total_s", ctypes.c_double),
                ("table_bytes", ctypes.c_int64), ("kappa", ctypes.c_int32), ("threads_used", ctypes.c_int32)]

SC_PROOF_BYTES = 8 + 16 + MAX_VARS * 3 * 16 + MAX_VARS * 16 + 32   # mle_sc_proof_t

def _parse_sc_proof(buf: np.ndarray) -> Tuple[int, Ext, List[Tuple[Ext, Ext, Ext]], List[Ext], Ext, Ext]:
    raw = buf.view(np.uint8)
    n = int(np.frombuffer(raw[0:4].tobytes(), dtype=np.int32)[0])
    w = raw[8:].view(np.uint64).reshape(-1, 2)  # ext words starting at byte 8
    claim = (int(w[0, 0]), int(w[0, 1]))
    msgs = [tuple((int(w[1 + 3 * i + k, 0]), int(w[1 + 3 * i + k, 1])) for k in range(3)) for i in range(n)]
    chal0 = 1 + 3 * MAX_VARS
    chals = [(int(w[chal0 + i, 0]), int(w[chal0 + i, 1])) for i in range(n)]
    fin = chal0 + MAX_VARS
    return n, claim, msgs, chals, (int(w[fin, 0]), int(w[fin, 1])), (int(w[fin + 1, 0]), int(w[fin + 1, 1]))

def build_mle_lib(force: bool = False) -> Dict:
    """Compile kernels/cpu/{mle,sumfold,sha256}.c into edgecert_mle.dll (MinGW-w64 gcc 6.2)."""
    DLL_PATH.parent.mkdir(parents=True, exist_ok=True)
    deps = [CPU_DIR / s for s in DLL_SOURCES + DLL_HEADERS]
    stale = force or not DLL_PATH.exists() or any(d.stat().st_mtime > DLL_PATH.stat().st_mtime for d in deps)
    flag_str = " ".join(DLL_FLAGS)
    gcc = str(MINGW_BIN / "gcc.exe") if MINGW_BIN.exists() else "gcc"
    cmd = [gcc] + DLL_FLAGS + [f'-DMLE_BUILD_FLAGS="{flag_str}"', "-o", str(DLL_PATH)] + \
        [str(CPU_DIR / s) for s in DLL_SOURCES]
    if stale:
        env = dict(os.environ)
        env["PATH"] = str(MINGW_BIN) + os.pathsep + env.get("PATH", "")
        r = subprocess.run(cmd, cwd=str(CPU_DIR), env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("edgecert_mle.dll build failed:\n" + r.stdout + r.stderr)
    return {"dll": str(DLL_PATH), "compiler": "gcc 6.2.0 (MinGW-w64, Vivado tps)", "flags": flag_str,
            "command": subprocess.list2cmdline(cmd), "rebuilt": stale}

_LIB = None
_LIB_LOCK = threading.Lock()

class MleLib:
    """ctypes wrapper of kernels/cpu/build/edgecert_mle.dll."""

    def __init__(self, path: Optional[str] = None, build: bool = True):
        self.build_info = build_mle_lib() if build else {"dll": str(DLL_PATH)}
        L = ctypes.CDLL(str(path or DLL_PATH))
        V, I, I64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64
        L.mle_build_info.restype = ctypes.c_char_p
        L.mle_lift.argtypes = [V, I, I64, I64, V]
        L.mle_fold_axis.argtypes = [V, I, I64, I64, I64, I64, V, I, V, I]
        L.mle_sumcheck.argtypes = [V, V, I, V, I, V, I, I, I, I, V]
        L.mm_prove.argtypes = [V, I, I64, I64, I64, I64, V, I, I64, I64, I64, V, I, V, I, V, I, V, I, I, V,
                               ctypes.POINTER(MmTimes), V, V]
        L.mlev_eq_table.argtypes = [V, I, V]
        L.mlev_bilinear.argtypes = [V, I, I64, I64, I64, I64, V, V, I, V]
        L.mle_scratch_peak.restype = I64
        L.mle_scratch_current.restype = I64
        for fn in ("mle_lift", "mle_fold_axis", "mle_sumcheck", "mm_prove", "mlev_eq_table", "mlev_bilinear"):
            getattr(L, fn).restype = I
        self.L = L
        self.flags = L.mle_build_info().decode()

    # -------------------------------------------------------------- raw helpers
    @staticmethod
    def _p(a: np.ndarray) -> int:
        return a.ctypes.data

    def lift(self, x: np.ndarray) -> np.ndarray:
        x1 = np.ascontiguousarray(x).reshape(-1)
        out = np.empty(x1.size, dtype=np.uint64)
        rc = self.L.mle_lift(self._p(x1), dtype_code(x1), x1.size, 1, self._p(out))
        assert rc == 0
        return out

    def fold_axis(self, M: np.ndarray, r: Sequence[Ext], nr: int, threads: int = 1) -> np.ndarray:
        """Logical (nfold x nkeep) view M -> (nkeep, 2) uint64 = M~(r, k)."""
        nfold, nkeep = M.shape
        sf, sk = elem_strides(M)
        ra = _ext_arr(r)
        out = np.empty((nkeep, 2), dtype=np.uint64)
        rc = self.L.mle_fold_axis(self._p(M), dtype_code(M), nfold, nkeep, sf, sk, self._p(ra), nr,
                                  self._p(out), threads)
        if rc < 0:
            raise RuntimeError(f"mle_fold_axis rc={rc}")
        return out

    def sumcheck(self, f: np.ndarray, g: np.ndarray, ctx: Optional[bytes] = None, label: Optional[bytes] = None,
                 engine: int = 0, threads: int = 1):
        """f, g: (2^n, 2) uint64 (copied).  Returns (claim, msgs, chals, final_f, final_g)."""
        n = f.shape[0].bit_length() - 1
        fc, gc = np.array(f, dtype=np.uint64, order="C"), np.array(g, dtype=np.uint64, order="C")
        buf = np.zeros(SC_PROOF_BYTES // 8, dtype=np.uint64)
        bind_ = 1 if ctx is not None else 0
        cb = ctx or b""
        lb = label or b""
        rc = self.L.mle_sumcheck(self._p(fc), self._p(gc), n, cb, len(cb), lb, len(lb), bind_, engine, threads,
                                 self._p(buf))
        if rc != 0:
            raise RuntimeError(f"mle_sumcheck rc={rc}")
        _, claim, msgs, chals, ff, fg = _parse_sc_proof(buf)
        return claim, msgs, chals, ff, fg

    def eq_table(self, r: Sequence[Ext]) -> np.ndarray:
        nr = len(r)
        ra = _ext_arr(r)
        out = np.empty((1 << nr, 2), dtype=np.uint64)
        rc = self.L.mlev_eq_table(self._p(ra), nr, self._p(out))
        assert rc == 0
        return out

    def bilinear(self, M: np.ndarray, eqR: np.ndarray, eqC: np.ndarray, threads: int = 1) -> Ext:
        R, C = M.shape
        sR, sC = elem_strides(M)
        out = np.zeros(2, dtype=np.uint64)
        rc = self.L.mlev_bilinear(self._p(M), dtype_code(M), R, C, sR, sC, self._p(eqR), self._p(eqC), threads,
                                  self._p(out))
        if rc != 0:
            raise RuntimeError(f"mlev_bilinear rc={rc}")
        return (int(out[0]), int(out[1]))

    def scratch_peak(self) -> int:
        return int(self.L.mle_scratch_peak())

    def scratch_reset(self):
        self.L.mle_scratch_reset()

def get_lib() -> MleLib:
    global _LIB
    with _LIB_LOCK:
        if _LIB is None:
            _LIB = MleLib()
        return _LIB

class CpuBackend:
    """kernels/cpu/mle.c through ctypes (the GIL is released during the call)."""
    name = "cpu"

    def __init__(self, lib: Optional[MleLib] = None):
        self.lib = lib or get_lib()

    def prove(self, inst: MatmulInstance, b: Optional[Binding] = None, threads: int = 1,
              want_vectors: bool = False):
        inst.check_shapes()
        T, K, N = inst.shape
        t0 = time.perf_counter()
        b = b or bind(inst)
        t1 = time.perf_counter()
        A, W = inst.A, inst.W
        sAt, sAk = elem_strides(A)
        sWk, sWn = elem_strides(W)
        r1, r2 = _ext_arr(b.r1), _ext_arr(b.r2)
        buf = np.zeros(SC_PROOF_BYTES // 8, dtype=np.uint64)
        tm = MmTimes()
        a_out = np.empty((K, 2), dtype=np.uint64) if want_vectors else None
        w_out = np.empty((K, 2), dtype=np.uint64) if want_vectors else None
        p = MleLib._p
        rc = self.lib.L.mm_prove(p(A), dtype_code(A), T, K, sAt, sAk, p(W), dtype_code(W), N, sWk, sWn,
                                 p(r1), b.t, p(r2), b.nb, b.ctx, len(b.ctx), b.label, len(b.label), threads,
                                 p(buf), ctypes.byref(tm), p(a_out) if want_vectors else None,
                                 p(w_out) if want_vectors else None)
        t2 = time.perf_counter()
        if rc != 0:
            raise RuntimeError(f"mm_prove({inst.layer_id}) rc={rc}")
        n, claim, msgs, chals, fa, fw = _parse_sc_proof(buf)
        pf = MatmulProof(inst.layer_id, T, K, N, n, claim, msgs, fa, fw, chals, backend=self.name)
        out = {"bind_s": t1 - t0, "fold_a_s": tm.fold_a_s, "fold_w_s": tm.fold_w_s, "sumcheck_s": tm.sumcheck_s,
               "core_s": tm.total_s, "wall_s": t2 - t0, "threads_used": tm.threads_used,
               "table_bytes": tm.table_bytes}
        if want_vectors:
            out["a"], out["w"] = _ext_list(a_out), _ext_list(w_out)
        return pf, out

# ================================================================ GPU backend (CuPy)
@dataclass
class DevMat:
    """Device operand: element (i, j) of instance b at arr[offset + b*bstride + i*s0 + j*s1]."""
    arr: object
    dtype_code: int
    offset: int
    bstride: int
    s0: int
    s1: int

class GpuBackend:
    name = "gpu"
    BLOCK = 256
    SC_BLOCK = 256
    MAX_LABEL = 96

    def __init__(self, device: int = 0):
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        import cupy as cp
        self.cp = cp
        self.device = device
        cp.cuda.Device(device).use()
        code = (CUDA_DIR / "sumfold.cu").read_text(encoding="utf-8") + "\n" + \
            (CUDA_DIR / "mle.cu").read_text(encoding="utf-8")
        self.mod = cp.RawModule(code=code, options=("-std=c++14",), backend="nvrtc")
        self.k = {n: self.mod.get_function(n) for n in ("mle_fold_i8", "mle_fold_u16", "mle_fold_ext",
                                                         "mm_sumcheck")}
        props = cp.cuda.runtime.getDeviceProperties(device)
        nm = props["name"]
        self.props = {"name": nm.decode() if isinstance(nm, bytes) else nm, "cc": f'{props["major"]}.{props["minor"]}',
                      "sm_count": props["multiProcessorCount"],
                      "global_mem_GiB": round(props["totalGlobalMem"] / 2**30, 1),
                      "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"), "cupy_index": device}
        self.kernel_attrs = {}
        for n, f in self.k.items():
            try:
                self.kernel_attrs[n] = {"num_regs": f.num_regs, "local_size_bytes": f.local_size_bytes}
            except Exception:  # pragma: no cover
                pass
        self.peak_pool_bytes = 0

    def to_device(self, a: np.ndarray):
        return self.cp.asarray(np.ascontiguousarray(a))

    def _note_pool(self):
        pool = self.cp.get_default_memory_pool()
        self.peak_pool_bytes = max(self.peak_pool_bytes, int(pool.total_bytes()))

    def _ptr(self, dm: DevMat) -> int:
        return int(dm.arr.data.ptr) + dm.offset * dm.arr.itemsize

    def _fold(self, dm: DevMat, fold_axis: int, nfold: int, nkeep: int, batch: int, chal, chal_bstride: int,
              lev_base: int, nr: int, out, out_bstride: int):
        """Partial evaluation along fold_axis of every instance -> out[b*out_bstride + k]."""
        cp, np_ = self.cp, np
        sf, sk = (dm.s0, dm.s1) if fold_axis == 0 else (dm.s1, dm.s0)
        kern = {MLE_I8: "mle_fold_i8", MLE_U16: "mle_fold_u16"}[dm.dtype_code]
        src_ptr, src_bstride, cur_sf, cur_sk, cur_n = self._ptr(dm), dm.bstride, sf, sk, nfold
        levels = [min(4, nr)] if nr <= 4 else [4] * (nr // 4) + ([nr % 4] if nr % 4 else [])
        lev = lev_base
        tmp_keep = []
        for pi, nlev in enumerate(levels):
            last = pi == len(levels) - 1
            nseg = (cur_n + (1 << nlev) - 1) >> nlev
            if last:
                dst, d_bstride, d_sf, d_sk = out, out_bstride, 0, 1
                assert nseg == 1
            else:
                dst = cp.empty((batch * nseg * nkeep, 2), dtype=cp.uint64)
                tmp_keep.append(dst)
                d_bstride, d_sf, d_sk = nseg * nkeep, nkeep, 1
            total = nseg * nkeep
            grid = (max(1, min(-(-total // self.BLOCK), 65535)), batch)
            seg_fastest = 1 if cur_sf == 1 else 0
            self.k[kern if pi == 0 else "mle_fold_ext"](
                grid, (self.BLOCK,),
                (np_.uint64(src_ptr), np_.int64(src_bstride), np_.int64(cur_n), np_.int64(nkeep), np_.int64(cur_sf),
                 np_.int64(cur_sk), chal, np_.int32(chal_bstride), np_.int32(lev), np_.int32(nlev), dst,
                 np_.int64(d_bstride), np_.int64(d_sf), np_.int64(d_sk), np_.int32(seg_fastest)))
            lev += nlev
            src_ptr, src_bstride, cur_sf, cur_sk, cur_n = int(dst.data.ptr), d_bstride, d_sf, d_sk, nseg
        return tmp_keep

    def prove_batch(self, insts: Sequence[MatmulInstance], dA: DevMat, dW: DevMat,
                    bindings: Optional[Sequence[Binding]] = None, pool=None):
        """Prove len(insts) same-shape matmuls; operands already resident on the device."""
        cp = self.cp
        B = len(insts)
        for x in insts:
            x.check_shapes()
        T, K, N = insts[0].shape
        t0 = time.perf_counter()
        if bindings is None:
            if pool is not None and B > 1:
                bindings = list(pool.map(bind, insts))
            else:
                bindings = [bind(x) for x in insts]
        t1 = time.perf_counter()
        t, nb, kappa = bindings[0].t, bindings[0].nb, bindings[0].kappa
        NK = 1 << kappa
        cb = max(1, t + nb)
        chal_h = np.zeros((B, cb, 2), dtype=np.uint64)
        meta_h = np.zeros(B * (32 + self.MAX_LABEL) + 4 * B, dtype=np.uint8)
        lens = np.zeros(B, dtype=np.int32)
        for i, bd in enumerate(bindings):
            for j, (c0, c1) in enumerate(bd.r1 + bd.r2):
                chal_h[i, j, 0], chal_h[i, j, 1] = c0, c1
            meta_h[32 * i:32 * i + 32] = np.frombuffer(bd.ctx, dtype=np.uint8)
            lb = bd.label[:self.MAX_LABEL]
            if len(bd.label) > self.MAX_LABEL:
                raise ValueError("label too long for the GPU transcript")
            o = 32 * B + self.MAX_LABEL * i
            meta_h[o:o + len(lb)] = np.frombuffer(lb, dtype=np.uint8)
            lens[i] = len(lb)
        meta_h[B * (32 + self.MAX_LABEL):] = lens.view(np.uint8)
        chal_d = cp.asarray(chal_h)
        meta_d = cp.asarray(meta_h)
        Fb = cp.empty((B * NK, 2), dtype=cp.uint64)
        Gb = cp.empty((B * NK, 2), dtype=cp.uint64)
        S1f = cp.empty((B * max(NK // 2, 1), 2), dtype=cp.uint64)
        S1g = cp.empty_like(S1f)
        S2f = cp.empty((B * max(NK // 4, 1), 2), dtype=cp.uint64)
        S2g = cp.empty_like(S2f)
        partials = cp.empty((3 * B, 2), dtype=cp.uint64)
        ostride = 3 + 4 * kappa
        out_d = cp.empty((B * ostride, 2), dtype=cp.uint64)
        e0 = cp.cuda.Event(); e1 = cp.cuda.Event(); e2 = cp.cuda.Event(); e3 = cp.cuda.Event()
        e0.record()
        keep = self._fold(dA, 0, T, K, B, chal_d, cb, 0, t, Fb, NK)
        e1.record()
        keep += self._fold(dW, 1, N, K, B, chal_d, cb, t, nb, Gb, NK)
        e2.record()
        mp = int(meta_d.data.ptr)
        self.k["mm_sumcheck"](
            (B,), (self.SC_BLOCK,),
            (Fb, Gb, np.int64(NK), np.int64(K), np.int32(kappa), S1f, S1g, np.int64(max(NK // 2, 1)), S2f, S2g,
             np.int64(max(NK // 4, 1)), np.uint64(mp), np.uint64(mp + 32 * B),
             np.uint64(mp + B * (32 + self.MAX_LABEL)), np.int32(self.MAX_LABEL), partials, out_d,
             np.int64(ostride)))
        e3.record()
        out_h = cp.asnumpy(out_d)      # synchronizes
        t2 = time.perf_counter()
        self._note_pool()
        del keep
        proofs = []
        for i, x in enumerate(insts):
            o = out_h[i * ostride:(i + 1) * ostride]
            e = lambda j: (int(o[j, 0]), int(o[j, 1]))  # noqa: E731
            msgs = [(e(1 + 3 * r), e(2 + 3 * r), e(3 + 3 * r)) for r in range(kappa)]
            chals = [e(1 + 3 * kappa + r) for r in range(kappa)]
            proofs.append(MatmulProof(x.layer_id, T, K, N, kappa, e(0), msgs, e(1 + 4 * kappa), e(2 + 4 * kappa),
                                      chals, backend=self.name))
        t3 = time.perf_counter()
        el = cp.cuda.get_elapsed_time
        tm = {"bind_s": t1 - t0, "fold_a_s": el(e0, e1) / 1e3, "fold_w_s": el(e1, e2) / 1e3,
              "sumcheck_s": el(e2, e3) / 1e3, "core_s": el(e0, e3) / 1e3, "wall_s": t3 - t0,
              "device_wall_s": t2 - t1, "batch": B}
        return proofs, tm

# ================================================================ verifier
class VerifyError(Exception):
    pass

def _py_eq(r: Sequence[Ext]):
    from ..verifier.mle import eq_table
    return eq_table(r)

def _py_bilinear(M: np.ndarray, eqR: Sequence[Ext], eqC: Sequence[Ext]) -> Ext:
    rows = M.tolist()
    acc0 = acc1 = 0
    for i, row in enumerate(rows):
        s0 = s1 = 0
        for j, m in enumerate(row):
            s0 += m * eqC[j][0]
            s1 += m * eqC[j][1]
        inner = (s0 % P, s1 % P)
        prod = emul(eqR[i], inner)
        acc0 += prod[0]
        acc1 += prod[1]
    return (acc0 % P, acc1 % P)

def verify_matmul(pf: MatmulProof, inst: MatmulInstance, engine: str = "c", threads: int = 1,
                  lib: Optional[MleLib] = None, timings: Optional[dict] = None) -> bool:
    """Independent verifier.  engine 'c': eq tables + contraction in the DLL (mlev_*);
    engine 'python': pure Python (small shapes).  Raises nothing; returns accept/reject."""
    try:
        _verify(pf, inst, engine, threads, lib, timings)
        return True
    except VerifyError:
        return False

def _verify(pf, inst, engine, threads, lib, timings):
    tm = timings if timings is not None else {}
    t0 = time.perf_counter()
    inst.check_shapes()
    T, K, N = inst.shape
    if (pf.T, pf.K, pf.N) != (T, K, N) or pf.layer_id != inst.layer_id:
        raise VerifyError("shape/layer mismatch")
    b = bind(inst)
    if pf.kappa != b.kappa or len(pf.msgs) != b.kappa:
        raise VerifyError("round count")
    t1 = time.perf_counter()
    if engine == "c":
        lib = lib or get_lib()
        eq1, eq2 = lib.eq_table(b.r1), lib.eq_table(b.r2)
        cv = lib.bilinear(inst.C, eq1, eq2, threads)
    else:
        eq1, eq2 = _py_eq(b.r1), _py_eq(b.r2)
        cv = _py_bilinear(inst.C, eq1, eq2)
    t2 = time.perf_counter()
    if cv != pf.claim:
        raise VerifyError("claim != C~(r1, r2)")
    tr = BoundSumcheckTranscript(b.kappa, pf.claim, b.ctx, b.label.decode())
    cur = pf.claim
    r3 = []
    for (s0, s1, s2) in pf.msgs:
        if eadd(s0, s1) != cur:
            raise VerifyError("round sum")
        r = tr.round(s0, s1, s2)
        r3.append(r)
        cur = interp_quadratic(s0, s1, s2, r)
    if emul(pf.final_a, pf.final_w) != cur:
        raise VerifyError("final product")
    t3 = time.perf_counter()
    if engine == "c":
        eq3 = lib.eq_table(r3)
        av = lib.bilinear(inst.A, eq1, eq3, threads)
        wv = lib.bilinear(inst.W, eq3, eq2, threads)
    else:
        eq3 = _py_eq(r3)
        av = _py_bilinear(inst.A, eq1, eq3)
        wv = _py_bilinear(inst.W, eq3, eq2)
    t4 = time.perf_counter()
    tm.update({"v_bind_s": t1 - t0, "v_claim_s": t2 - t1, "v_sumcheck_s": t3 - t2, "v_final_s": t4 - t3,
               "v_total_s": t4 - t0})
    if av != pf.final_a:
        raise VerifyError("a(r3) != A~(r1, r3)")
    if wv != pf.final_w:
        raise VerifyError("w(r3) != W~(r3, r2)")

# ================================================================ op counts
def fold_counts(m: int, nr: int) -> Dict[str, int]:
    """Ideal level-by-level folds of one zero-padded length-2^nr vector with m real entries."""
    base = ext = 0
    cur = m
    for level in range(nr):
        outs = (cur + 1) // 2
        if level == 0:
            base += outs
        else:
            ext += outs
        cur = outs
    return {"base_folds": base, "ext_folds": ext}

def matmul_opcount(T: int, K: int, N: int) -> Dict[str, int]:
    """prover/verifier operation counts for one matmul proof (see docs/notes/gpt2prove.md)."""
    t, nb, kappa = log2_ceil(T), log2_ceil(N), max(1, log2_ceil(K))
    fa, fw = fold_counts(T, t), fold_counts(N, nb)
    NK = 1 << kappa
    base_folds = K * (fa["base_folds"] + fw["base_folds"])
    ext_folds = K * (fa["ext_folds"] + fw["ext_folds"])
    sc_ext_mults = 5 * (NK - 1)
    fold_base_mults = 2 * base_folds + 3 * ext_folds
    sc_base_mults = 15 * (NK - 1)
    return {
        "T": T, "K": K, "N": N, "t": t, "nb": nb, "kappa": kappa,
        "fold_elements": T * K + K * N,
        "fold_base_folds": base_folds, "fold_ext_folds": ext_folds,
        "fold_base_mults": fold_base_mults,
        "sumcheck_ext_mults": sc_ext_mults, "sumcheck_base_mults": sc_base_mults,
        "prover_base_mults": fold_base_mults + sc_base_mults,
        "sumcheck_sha256_msgs": 2 + kappa,        # init, binding, one per round
        "challenge_sha256_msgs": 2 + t + nb,      # ProofTranscript init, absorb layer id, r1, r2
        "proof_bytes": 16 + 48 * kappa + 32,
        "digest_bytes_per_proof": T * K + (T * N) * 4,   # activations int8 + int32 output (digest(W) cached)
        "verifier_mac_int_x_ext": T * N + T * K + K * N,   # C~, A~, W~ contractions (2 base mults each)
        "verifier_eq_ext_mults": (1 << t) - 1 + (1 << nb) - 1 + NK - 1,
    }
