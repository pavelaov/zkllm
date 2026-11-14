"""Pure-Python emulation of the index and reduction logic of kernels/cuda/sumfold.cu.

This is a *transcription* of the CUDA control flow (grid-stride loops, per-thread lazy accumulators, warp shuffle-down trees, the shared-memory warp stage, the one-block partial reduction, ping-pong buffers and the host round loop) with exact field arithmetic from the golden model.  It lets us check, without a GPU, that

* every pair / quad index is visited exactly once for any (grid, block) shape, * the two-stage warp reduction combines exactly the per-thread partials, * the result is independent of block execution order (ping-pong), * the global in-place alternative is order-dependent (the reason for ping-pong).

The actual kernel source is additionally *executed* on the CPU through the SIMT shim (kernels/cuda/shim, backend "shim" in kernel_gpu_bench.py).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from edgecert.field import P, ZERO, eadd, emul, esub, edouble, fadd, reduce128  # noqa: E402
from edgecert.sumfold import Transcript  # noqa: E402

WARP = 32
MASK64 = (1 << 64) - 1

Ext = Tuple[int, int]

def block_order(grid: int, order: str) -> List[int]:
    if order == "asc":
        return list(range(grid))
    if order == "desc":
        return list(range(grid - 1, -1, -1))
    if order == "oddfirst":
        return [b for b in range(grid) if b % 2] + [b for b in range(grid) if b % 2 == 0]
    raise ValueError(order)

# --------------------------------------------------------------- reductions
def _lazy_reduce(acc: int) -> int:
    """acc6 lazy accumulator: exact 128-bit sum, then gl_reduce128."""
    assert 0 <= acc < (1 << 128), "lazy accumulator overflow"
    return reduce128(acc)

def warp_reduce(vals: Sequence[int]) -> List[int]:
    """__shfl_down_sync tree on 32 lanes; all lanes update simultaneously."""
    v = list(vals)
    assert len(v) == WARP
    off = 16
    while off > 0:
        v = [fadd(v[l], v[l + off] if l + off < WARP else v[l]) for l in range(WARP)]
        off >>= 1
    return v

def block_reduce(thread_acc: List[List[int]], block: int) -> List[Ext]:
    """block_reduce_store: thread_acc[t][k] = exact integer sums (k = 0..5)."""
    assert block % WARP == 0 and block <= 1024
    nwarps = block // WARP
    v = [[_lazy_reduce(thread_acc[t][k]) for k in range(6)] for t in range(block)]
    smem = [[0] * 6 for _ in range(32)]
    for w in range(nwarps):
        for k in range(6):
            lanes = warp_reduce([v[w * WARP + l][k] for l in range(WARP)])
            smem[w][k] = lanes[0]  # lane 0 writes
    # second stage: executed by every warp; thread 0 (warp 0, lane 0) stores
    out = []
    for k in range(6):
        lanes = warp_reduce([smem[l][k] if l < nwarps else 0 for l in range(WARP)])
        out.append(lanes[0])
    return [(out[0], out[1]), (out[2], out[3]), (out[4], out[5])]

def _acc_pair(acc: List[int], f0, f1, g0, g1) -> None:
    p0 = emul(f0, g0)
    p1 = emul(f1, g1)
    p2 = emul(esub(edouble(f1), f0), esub(edouble(g1), g0))
    for k, x in enumerate((p0[0], p0[1], p1[0], p1[1], p2[0], p2[1])):
        acc[k] += x

def _fold(v0: Ext, v1: Ext, r: Ext) -> Ext:
    return eadd(v0, emul(r, esub(v1, v0)))

# ------------------------------------------------------------------ kernels
def k_eval(f, g, npairs, grid, block, order, partials, visits=None):
    stride = grid * block
    for b in block_order(grid, order):
        accs = []
        for t in range(block):
            acc = [0] * 6
            j = b * block + t
            while j < npairs:
                if visits is not None:
                    visits[j] += 1
                _acc_pair(acc, f[2 * j], f[2 * j + 1], g[2 * j], g[2 * j + 1])
                j += stride
            accs.append(acc)
        partials[3 * b:3 * b + 3] = block_reduce(accs, block)

def k_fold_eval(fs, gs, fd, gd, nquads, r, grid, block, order, partials, visits=None):
    """Threads of a block are emulated one after another (worst case for hazards is the
    block order; within the ping-pong design no ordering can matter)."""
    stride = grid * block
    for b in block_order(grid, order):
        accs = []
        for t in range(block):
            acc = [0] * 6
            k = b * block + t
            while k < nquads:
                if visits is not None:
                    visits[k] += 1
                a0 = _fold(fs[4 * k], fs[4 * k + 1], r)
                a1 = _fold(fs[4 * k + 2], fs[4 * k + 3], r)
                b0 = _fold(gs[4 * k], gs[4 * k + 1], r)
                b1 = _fold(gs[4 * k + 2], gs[4 * k + 3], r)
                fd[2 * k], fd[2 * k + 1], gd[2 * k], gd[2 * k + 1] = a0, a1, b0, b1
                _acc_pair(acc, a0, a1, b0, b1)
                k += stride
            accs.append(acc)
        partials[3 * b:3 * b + 3] = block_reduce(accs, block)

def k_fold(fs, gs, fd, gd, npairs, r, grid, block, order):
    stride = grid * block
    for b in block_order(grid, order):
        for t in range(block):
            j = b * block + t
            while j < npairs:
                fd[j] = _fold(fs[2 * j], fs[2 * j + 1], r)
                gd[j] = _fold(gs[2 * j], gs[2 * j + 1], r)
                j += stride

def k_reduce_partials(partials, nblocks, block) -> List[Ext]:
    accs = []
    for t in range(block):
        acc = [0] * 6
        for b in range(t, nblocks, block):
            for k in range(3):
                acc[2 * k] += partials[3 * b + k][0]
                acc[2 * k + 1] += partials[3 * b + k][1]
        accs.append(acc)
    return block_reduce(accs, block)

# ---------------------------------------------------------------- host flow
def grid_for(work: int, block: int, grid_max: int) -> int:
    return max(1, min(-(-work // block), grid_max))

def emu_prove(f: Sequence[Ext], g: Sequence[Ext], variant: str = "fused", block: int = 64,
              grid_max: int = 3, order: str = "asc", inplace: bool = False, check_visits: bool = True):
    """Run the host round loop of kernel_gpu_bench.prove_device on the emulated kernels.

    inplace=True emulates the rejected design (dst = src) to demonstrate the hazard.
    Returns dict(claim, msgs, challenges, final_f, final_g) with Ext tuples.
    """
    N = len(f)
    n = N.bit_length() - 1
    assert N == 1 << n and n >= 1
    A_f, A_g = list(f), list(g)
    B_f, B_g = ([ZERO] * max(N // 2, 1), [ZERO] * max(N // 2, 1))
    if inplace:
        B_f, B_g = A_f, A_g
    src, dst = (A_f, A_g), (B_f, B_g)
    partials = [ZERO] * (3 * grid_max)
    msgs, chal = [], []
    tr = None
    claim = None

    def reduce_round(grid):
        return k_reduce_partials(partials, grid, block)

    if variant == "fused":
        npairs = N // 2
        grid = grid_for(npairs, block, grid_max)
        visits = [0] * npairs
        k_eval(src[0], src[1], npairs, grid, block, order, partials, visits)
        assert not check_visits or all(v == 1 for v in visits)
        s = reduce_round(grid)
        claim = eadd(s[0], s[1])
        tr = Transcript(n, claim)
        r = tr.round(*s)
        msgs.append(tuple(s)); chal.append(r)
        for i in range(1, n):
            nquads = (N >> (i - 1)) // 4
            grid = grid_for(nquads, block, grid_max)
            visits = [0] * nquads
            k_fold_eval(src[0], src[1], dst[0], dst[1], nquads, r, grid, block, order, partials, visits)
            assert not check_visits or all(v == 1 for v in visits)
            s = reduce_round(grid)
            r = tr.round(*s)
            msgs.append(tuple(s)); chal.append(r)
            src, dst = dst, src
        k_fold(src[0], src[1], dst[0], dst[1], 1, r, 1, block, order)
        final = (dst[0][0], dst[1][0])
    elif variant == "unfused":
        for i in range(n):
            npairs = (N >> i) // 2
            grid = grid_for(npairs, block, grid_max)
            k_eval(src[0], src[1], npairs, grid, block, order, partials)
            s = reduce_round(grid)
            if i == 0:
                claim = eadd(s[0], s[1])
                tr = Transcript(n, claim)
            r = tr.round(*s)
            msgs.append(tuple(s)); chal.append(r)
            k_fold(src[0], src[1], dst[0], dst[1], npairs, r, grid, block, order)
            src, dst = dst, src
        final = (src[0][0], src[1][0])
    else:
        raise ValueError(variant)
    return {"claim": claim, "msgs": msgs, "challenges": chal, "final_f": final[0], "final_g": final[1]}
