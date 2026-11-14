"""CertSched DAGs (contribution C3).

For a decoder-only transformer shape and a workload (prefill length, generated tokens) this module builds three dependency graphs over one shared node set:

* inference DAG   : tensors and the deterministic integer ops that produce them;
* proof DAG       : per-group proof tasks (matmul / Hadamard sumchecks, lookup and
                    range arguments, openings) and the tensors each one reads;
* commit/challenge: commit(witness segment) -> challenge derivation -> every
                    task or helper derivation that uses that challenge.

The schedule graph is the union of the three. Nothing here executes a proof; the graphs are the input of the legality checker and the policies in ``edgecert.runtime.scheduler``.

Byte model: every witness element is a Goldilocks element (8 B); post-challenge lookup helpers (LogUp inverses) are quadratic-extension elements (16 B); weights stay in compact int8 (1 B) because they are streamed and lifted to the field on the fly. A requantised tensor with E elements carries its value column y (E x 8 B), its remainder limbs (range_limbs x E x 8 B) and one helper column per range-checked column ((1 + range_limbs) x E x 16 B). These are design parameters, not measurements.

Protocol modes:
* GLOBAL    : one Fiat-Shamir transcript; every base witness is committed before
              the lookup challenge beta; all helpers before the sumcheck
              challenge gamma. Compatible with backends that need all witness
              commitments up front (conservative default).
* LAYERWISE : one transcript segment per (segment, layer) group, chained in a
              fixed order. Legal only if the backend supports per-group
              composition (an explicit proof obligation, see docs/notes/sched.md).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------- shapes

@dataclass(frozen=True)
class ModelShape:
    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    vocab: int
    gated_mlp: bool
    qk_norm: bool
    tied_embeddings: bool
    native_ctx: int
    learned_pos: bool = False
    sliding_window: int = 0      # 0: every layer is global
    global_every: int = 1        # layer i is global iff (i + 1) % global_every == 0
    source: str = ""
    sandwich_norm: bool = False  # extra post-attention / post-FFN norms (Gemma 3)

    def is_local(self, layer: int) -> bool:
        return self.sliding_window > 0 and (layer + 1) % self.global_every != 0

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    def layer_params(self) -> int:
        d, f = self.d_model, self.d_ff
        attn = d * self.q_dim + 2 * d * self.kv_dim + self.q_dim * d
        mlp = (3 if self.gated_mlp else 2) * d * f
        return attn + mlp

    def n_params(self) -> int:
        emb = self.vocab * self.d_model
        pos = self.native_ctx * self.d_model if self.learned_pos else 0
        head = 0 if self.tied_embeddings else self.vocab * self.d_model
        return emb + pos + head + self.n_layers * self.layer_params()

# Fallback table. get_shape() prefers src/edgecert/compiler/model_shapes.py (the shared source of truth, see COMPILER_NAMES) and only uses this table if that import fails. Values are the public configuration fields of each model card (REPORTED).
LOCAL_SHAPES: Dict[str, ModelShape] = {
    "gpt2-124m": ModelShape(
        "gpt2-124m", n_layers=12, d_model=768, n_heads=12, n_kv_heads=12, head_dim=64,
        d_ff=3072, vocab=50257, gated_mlp=False, qk_norm=False, tied_embeddings=True,
        native_ctx=1024, learned_pos=True,
        source="GPT-2 small config: n_layer 12, n_embd 768, n_head 12, n_inner 3072, vocab 50257"),
    "qwen3-0.6b": ModelShape(
        "qwen3-0.6b", n_layers=28, d_model=1024, n_heads=16, n_kv_heads=8, head_dim=128,
        d_ff=3072, vocab=151936, gated_mlp=True, qk_norm=True, tied_embeddings=True,
        native_ctx=32768,
        source="Qwen3-0.6B config: 28 layers, hidden 1024, 16 q / 8 kv heads, head_dim 128, "
               "intermediate 3072, vocab 151936, tied embeddings"),
    "gemma3-1b": ModelShape(
        "gemma3-1b", n_layers=26, d_model=1152, n_heads=4, n_kv_heads=1, head_dim=256,
        d_ff=6912, vocab=262144, gated_mlp=True, qk_norm=True, tied_embeddings=True,
        native_ctx=32768, sliding_window=512, global_every=6, sandwich_norm=True,
        source="Gemma-3-1B config: 26 layers, hidden 1152, 4 q / 1 kv heads, head_dim 256, "
               "intermediate 6912, vocab 262144, sliding window 512, 5 local : 1 global"),
    "llama3-8b": ModelShape(
        "llama3-8b", n_layers=32, d_model=4096, n_heads=32, n_kv_heads=8, head_dim=128,
        d_ff=14336, vocab=128256, gated_mlp=True, qk_norm=False, tied_embeddings=False,
        native_ctx=8192,
        source="8B-class reference (Meta-Llama-3-8B config: 32 layers, hidden 4096, 32 q / 8 kv "
               "heads, intermediate 14336, vocab 128256)"),
}

MODEL_ORDER = ["gpt2-124m", "qwen3-0.6b", "gemma3-1b", "llama3-8b"]

COMPILER_NAMES = {"gpt2-124m": "gpt2_124m", "qwen3-0.6b": "qwen3_0.6b",
                  "gemma3-1b": "gemma3_1b", "llama3-8b": "llama3_8b"}

def get_shape(name: str) -> ModelShape:
    """Shape from src/edgecert/compiler/model_shapes.py (shared table), else local fallback."""
    try:
        from edgecert.compiler import model_shapes as ms  # type: ignore
        table = getattr(ms, "MODELS", None)
        key = COMPILER_NAMES.get(name, name)
        if isinstance(table, dict) and key in table:
            return _from_compiler(table[key], name)
    except Exception:  # pragma: no cover - fallback only
        pass
    return LOCAL_SHAPES[name]

def _from_compiler(s, name: str) -> ModelShape:
    """Map compiler.model_shapes.ModelShape onto the fields the scheduler needs."""
    unver = tuple(getattr(s, "unverified_fields", ()) or ())
    return ModelShape(
        name=name, n_layers=s.n_layers, d_model=s.d_model, n_heads=s.n_heads,
        n_kv_heads=s.n_kv_heads, head_dim=s.head_dim, d_ff=s.d_ff, vocab=s.vocab,
        gated_mlp=(s.mlp != "gelu"), qk_norm=bool(s.qk_norm), tied_embeddings=bool(s.tied_embeddings),
        native_ctx=int(s.ctx), learned_pos=(s.pos == "learned"),
        sliding_window=int(s.sliding_window or 0), global_every=int(s.global_every or 1),
        sandwich_norm=bool(getattr(s, "sandwich_norm", False)),
        source=(f"edgecert.compiler.model_shapes.{s.name} (hf_id {getattr(s, 'hf_id', '?')}; "
                f"unverified fields: {', '.join(unver) if unver else 'none'})"))

# ------------------------------------------------------------------ workload

@dataclass(frozen=True)
class Workload:
    prefill: int
    generated: int
    decode_block: int = 0     # 0: all decode steps form one segment

    def segments(self) -> List[Tuple[int, int, int]]:
        """(pos0, rows, output_rows) per segment.

        Segment 0 is the prefill (rows = prefill, its last row yields token 1).
        Decode segments process generated tokens 1..G-1 (the last generated token
        is never fed back), each row yields the next token.
        """
        segs = [(0, self.prefill, 1)]
        dec = self.generated - 1
        if dec > 0:
            blk = self.decode_block if self.decode_block > 0 else dec
            p = self.prefill
            left = dec
            while left > 0:
                r = min(blk, left)
                segs.append((p, r, r))
                p += r
                left -= r
        return segs

    @property
    def total_positions(self) -> int:
        return self.prefill + max(self.generated - 1, 0)

@dataclass(frozen=True)
class Encoding:
    base_bytes: int = 8          # Goldilocks element
    ext_bytes: int = 16          # quadratic extension element
    weight_bytes: int = 1        # int8 weights, lifted on the fly
    range_limbs: int = 2         # 16-bit requant remainder as 2 x 8-bit limbs
    logit_limbs: int = 4         # 32-bit argmax gap as 4 x 8-bit limbs
    protocol: str = "GLOBAL"     # GLOBAL | LAYERWISE
    head_split: bool = True      # attention internals as one tensor per head

# --------------------------------------------------------------------- nodes

@dataclass
class Tensor:
    id: str
    group: Tuple[int, int]            # (segment, layer); layer -1 embed, n_layers final
    role: str                         # y | aux | lk | helper | weight | token
    elems: int
    elem_bytes: int
    committed: bool
    producer: Optional[str] = None    # op id (inference op or helper derivation)
    anchor: bool = False              # recompute anchor (layer input checkpoint, token)
    static: bool = False              # model weight, device-resident, model-committed

    @property
    def nbytes(self) -> int:
        return self.elems * self.elem_bytes

@dataclass
class Op:
    """Deterministic integer op (inference) or helper derivation (proof side)."""
    id: str
    group: Tuple[int, int]
    kind: str
    inputs: List[str]
    outputs: List[str]
    macs: int = 0
    challenge: Optional[str] = None   # derive ops: the challenge they use
    deterministic: bool = True
    work: int = 0                     # proof-side work elements of its sumcheck (0: default)

@dataclass
class ProofTask:
    id: str
    group: Tuple[int, int]
    kind: str                          # sc_matmul | sc_hadamard | lookup | open
    reads: List[str]
    challenges: List[str]
    work: int                          # abstract work elements (see cost model)
    target: Optional[str] = None       # open: the committed tensor opened

@dataclass
class Challenge:
    id: str
    kind: str                          # beta | gamma
    depends: List[str]                 # committed tensor ids (commitments) it binds
    group: Optional[Tuple[int, int]] = None

@dataclass
class LookupReq:
    id: str
    group: Tuple[int, int]
    columns: List[str]
    elems: int                         # rows per column
    table: int                         # table size
    helper: str = ""

# ----------------------------------------------------------------------- DAG

@dataclass
class CertDAG:
    shape: ModelShape
    workload: Workload
    enc: Encoding
    tensors: Dict[str, Tensor] = field(default_factory=dict)
    ops: Dict[str, Op] = field(default_factory=dict)           # inference ops (topological)
    derive_ops: Dict[str, Op] = field(default_factory=dict)    # helper derivations
    tasks: Dict[str, ProofTask] = field(default_factory=dict)  # canonical order
    challenges: Dict[str, Challenge] = field(default_factory=dict)
    lookups: Dict[str, LookupReq] = field(default_factory=dict)
    groups: List[Tuple[int, int]] = field(default_factory=list)
    transcript: List[Tuple[str, str]] = field(default_factory=list)  # canonical order

    # ---- derived views
    def producer(self, tid: str) -> Optional[Op]:
        p = self.tensors[tid].producer
        if p is None:
            return None
        return self.ops.get(p) or self.derive_ops.get(p)

    def committed_tensors(self) -> List[str]:
        return [t.id for t in self.tensors.values() if t.committed]

    def witness_bytes(self, roles: Optional[Iterable[str]] = None) -> int:
        rs = set(roles) if roles else None
        return sum(t.nbytes for t in self.tensors.values()
                   if t.committed and (rs is None or t.role in rs))

    def static_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors.values() if t.static)

    def inference_edges(self) -> List[Tuple[str, str]]:
        e = []
        for op in self.ops.values():
            e += [(i, op.id) for i in op.inputs] + [(op.id, o) for o in op.outputs]
        return e

    def proof_edges(self) -> List[Tuple[str, str]]:
        e = []
        for op in self.derive_ops.values():
            e += [(i, op.id) for i in op.inputs] + [(op.id, o) for o in op.outputs]
        for t in self.tasks.values():
            e += [(r, t.id) for r in t.reads]
        return e

    def commit_challenge_edges(self) -> List[Tuple[str, str]]:
        e = []
        for c in self.challenges.values():
            e += [("C:" + d, c.id) for d in c.depends]
        for t in self.tasks.values():
            e += [(c, t.id) for c in t.challenges]
        for op in self.derive_ops.values():
            if op.challenge:
                e.append((op.challenge, op.id))
        return e

    def summary(self) -> dict:
        roles: Dict[str, int] = {}
        for t in self.tensors.values():
            if t.committed:
                roles[t.role] = roles.get(t.role, 0) + t.nbytes
        return {
            "model": self.shape.name,
            "prefill": self.workload.prefill,
            "generated": self.workload.generated,
            "protocol": self.enc.protocol,
            "n_tensors": len(self.tensors),
            "n_committed": sum(1 for t in self.tensors.values() if t.committed),
            "n_ops": len(self.ops),
            "n_derive_ops": len(self.derive_ops),
            "n_tasks": len(self.tasks),
            "n_challenges": len(self.challenges),
            "n_groups": len(self.groups),
            "witness_bytes": self.witness_bytes(),
            "witness_bytes_by_role": roles,
            "static_weight_bytes": self.static_bytes(),
            "total_macs": sum(o.macs for o in self.ops.values()),
            "beyond_native_ctx": self.workload.total_positions > self.shape.native_ctx,
            "native_ctx": self.shape.native_ctx,
            "shape_source": self.shape.source,
        }

def _score_elems(pos0: int, rows: int, window: int) -> int:
    """sum_{p=pos0}^{pos0+rows-1} min(p+1, window) (window 0 = unbounded, causal)."""
    a, b = pos0 + 1, pos0 + rows          # sum of min(q, w) for q in [a, b]
    if window <= 0 or b <= window:
        return (a + b) * rows // 2
    if a > window:
        return window * rows
    s1 = (a + window) * (window - a + 1) // 2
    return s1 + window * (b - window)

class _Builder:
    def __init__(self, shape: ModelShape, wl: Workload, enc: Encoding):
        self.s, self.wl, self.enc = shape, wl, enc
        self.dag = CertDAG(shape, wl, enc)
        self.group_ops: Dict[Tuple[int, int], List[str]] = {}

    # -- primitive constructors
    def tensor(self, tid, group, role, elems, eb, committed, producer=None, anchor=False, static=False):
        t = Tensor(tid, group, role, int(elems), eb, committed, producer, anchor, static)
        self.dag.tensors[tid] = t
        return tid

    def op(self, oid, group, kind, inputs, macs=0):
        o = Op(oid, group, kind, list(inputs), [], int(macs))
        self.dag.ops[oid] = o
        self.group_ops.setdefault(group, []).append(oid)
        return o

    def weight(self, wid, elems):
        if wid not in self.dag.tensors:
            self.tensor(wid, (-1, -1), "weight", elems, self.enc.weight_bytes, False, static=True)
        return wid

    def lookup(self, group, name, columns, elems, table, n_helper_cols):
        lid = f"lk:{name}"
        self.dag.lookups[lid] = LookupReq(lid, group, list(columns), int(elems), int(table))
        hid = f"h:{name}"
        self.tensor(hid, group, "helper", elems * n_helper_cols, self.enc.ext_bytes, True,
                    producer=f"d:{name}")
        self.dag.lookups[lid].helper = hid
        return lid

    def requant(self, group, name, elems, kind, inputs, macs, limbs=None, anchor=False):
        """Requantised output bundle: y, remainder limbs, range-check helper."""
        limbs = self.enc.range_limbs if limbs is None else limbs
        o = self.op(f"op:{name}", group, kind, inputs, macs)
        y = self.tensor(f"y:{name}", group, "y", elems, self.enc.base_bytes, True, o.id, anchor=anchor)
        aux = self.tensor(f"a:{name}", group, "aux", elems * limbs, self.enc.base_bytes, True, o.id)
        o.outputs = [y, aux]
        self.lookup(group, name, [y, aux], elems, 256, 1 + limbs)
        return y

    def table_lookup(self, group, name, elems, kind, inputs, macs, table=1 << 16, anchor=False):
        """Lookup-table op (exp, activation, embedding): output + helper."""
        o = self.op(f"op:{name}", group, kind, inputs, macs)
        y = self.tensor(f"y:{name}", group, "lk", elems, self.enc.base_bytes, True, o.id, anchor=anchor)
        o.outputs = [y]
        self.lookup(group, name, list(inputs[:1]) + [y], elems, table, 1)
        return y

    # -- model
    def build(self) -> CertDAG:
        s, enc = self.s, self.enc
        d, f, dh = s.d_model, s.d_ff, s.head_dim
        qd, kd = s.q_dim, s.kv_dim
        L = s.n_layers
        kv_hist: Dict[int, List[Tuple[str, str, int]]] = {l: [] for l in range(L)}  # (k, v, rows)
        prev_tok: Optional[str] = None
        segs = self.wl.segments()
        for si, (pos0, rows, out_rows) in enumerate(segs):
            # ---- embedding group
            g = (si, -1)
            if prev_tok is None:
                tok = self.tensor(f"tok:{si}", g, "token", rows, enc.base_bytes, True,
                                  producer=None, anchor=True)
            else:
                                # autoregressive feed: the decode segment's tokens are argmax outputs of the previous segment (within-block feedback is internal to the block op)
                fo = self.op(f"op:{si}.feed", g, "feed", [prev_tok], 0)
                tok = self.tensor(f"tok:{si}", g, "token", rows, enc.base_bytes, True,
                                  producer=fo.id, anchor=True)
                fo.outputs = [tok]
            w_emb = self.weight("W:emb", s.vocab * d + (s.native_ctx * d if s.learned_pos else 0))
            x = self.table_lookup(g, f"{si}.emb", rows * d, "embed", [tok, w_emb], rows * d,
                                  table=s.vocab, anchor=True)
            for l in range(L):
                g = (si, l)
                p = f"{si}.{l}"
                xn1 = self.requant(g, p + ".xn1", rows * d, "norm", [x], rows * d * 4)
                q = self.requant(g, p + ".q", rows * qd, "matmul", [xn1, self.weight(f"W:{l}.q", d * qd)],
                                 rows * d * qd)
                k = self.requant(g, p + ".k", rows * kd, "matmul", [xn1, self.weight(f"W:{l}.k", d * kd)],
                                 rows * d * kd)
                v = self.requant(g, p + ".v", rows * kd, "matmul", [xn1, self.weight(f"W:{l}.v", d * kd)],
                                 rows * d * kd)
                if s.qk_norm:
                    q = self.requant(g, p + ".qn", rows * qd, "norm", [q], rows * qd * 4)
                    k = self.requant(g, p + ".kn", rows * kd, "norm", [k], rows * kd * 4)
                kv_hist[l].append((k, v, rows))
                window = s.sliding_window if s.is_local(l) else 0
                sc_el = _score_elems(pos0, rows, window)
                ctx_keys = pos0 + rows if window <= 0 else min(pos0 + rows, window + rows)
                ks = [kk for (kk, _, _) in kv_hist[l]]
                vs = [vv for (_, vv, _) in kv_hist[l]]
                heads = list(range(s.n_heads)) if enc.head_split else [None]
                hcount = 1 if enc.head_split else s.n_heads
                o_parts = []
                for j in heads:
                    hp = p + (f".h{j}" if j is not None else ".hall")
                    S = self.requant(g, hp + ".s", sc_el * hcount, "attn_score", [q] + ks,
                                     sc_el * dh * hcount)
                    E = self.table_lookup(g, hp + ".e", sc_el * hcount, "softmax_exp", [S],
                                          sc_el * hcount)
                    P = self.requant(g, hp + ".p", sc_el * hcount, "softmax_norm", [E], sc_el * hcount)
                    O = self.requant(g, hp + ".o", rows * dh * hcount, "attn_pv", [P] + vs,
                                     sc_el * dh * hcount)
                    # per-head sumcheck work: q_j + K_j (all keys) + S_j, and P_j + V_j + o_j
                    self.dag.ops[f"op:{hp}.s"].work = hcount * (rows * dh + ctx_keys * dh + sc_el)
                    self.dag.ops[f"op:{hp}.o"].work = hcount * (sc_el + ctx_keys * dh + rows * dh)
                    o_parts.append(O)
                ao = self.requant(g, p + ".ao", rows * d, "matmul",
                                  o_parts + [self.weight(f"W:{l}.o", qd * d)], rows * qd * d)
                if s.sandwich_norm:   # post-attention norm (Gemma 3)
                    ao = self.requant(g, p + ".pn1", rows * d, "norm", [ao], rows * d * 4)
                                # mid-layer residual is a second recompute anchor so that MLP tensors can be recomputed without re-running the attention block
                x2 = self.requant(g, p + ".x2", rows * d, "residual", [x, ao], rows * d, anchor=True)
                xn2 = self.requant(g, p + ".xn2", rows * d, "norm", [x2], rows * d * 4)
                u = self.requant(g, p + ".up", rows * f, "matmul", [xn2, self.weight(f"W:{l}.up", d * f)],
                                 rows * d * f)
                if s.gated_mlp:
                    gt = self.requant(g, p + ".gate", rows * f, "matmul",
                                      [xn2, self.weight(f"W:{l}.gate", d * f)], rows * d * f)
                    a = self.table_lookup(g, p + ".act", rows * f, "activation", [gt], rows * f)
                    m = self.requant(g, p + ".mul", rows * f, "hadamard", [a, u], rows * f)
                else:
                    m = self.table_lookup(g, p + ".act", rows * f, "activation", [u], rows * f)
                dn = self.requant(g, p + ".down", rows * d, "matmul", [m, self.weight(f"W:{l}.down", f * d)],
                                  rows * f * d)
                if s.sandwich_norm:   # post-FFN norm (Gemma 3)
                    dn = self.requant(g, p + ".pn2", rows * d, "norm", [dn], rows * d * 4)
                x = self.requant(g, p + ".res", rows * d, "residual", [x2, dn], rows * d, anchor=True)
            # ---- final group: norm + LM head + argmax on output rows only
            g = (si, L)
            p = f"{si}.fin"
            xf = self.requant(g, p + ".xf", out_rows * d, "norm", [x], out_rows * d * 4)
            w_head = self.weight("W:emb", 0) if s.tied_embeddings else self.weight("W:head", s.vocab * d)
            lg = self.requant(g, p + ".logits", out_rows * s.vocab, "lmhead", [xf, w_head],
                              out_rows * d * s.vocab, limbs=enc.logit_limbs)
            am = self.op(f"op:{p}.argmax", g, "argmax", [lg], out_rows * s.vocab)
            nt = self.tensor(f"y:{p}.next", g, "y", out_rows, enc.base_bytes, True, am.id)
            am.outputs = [nt]
            prev_tok = nt
        self.dag.groups = self._group_order(len(segs), L)
        self._derive_ops()
        self._proof_tasks()
        self._challenges_and_transcript()
        return self.dag

    @staticmethod
    def _group_order(n_seg: int, L: int) -> List[Tuple[int, int]]:
        return [(si, l) for si in range(n_seg) for l in range(-1, L + 1)]

    def _derive_ops(self):
        for lk in self.dag.lookups.values():
            name = lk.id[3:]
            op = Op(f"d:{name}", lk.group, "derive", list(lk.columns), [lk.helper],
                    macs=self.dag.tensors[lk.helper].elems * 3)
            self.dag.derive_ops[op.id] = op

    def _proof_tasks(self):
        """Canonical per-group proof tasks, openings placed after their last involvement."""
        dag = self.dag
        s = self.s
        chal_of = self._challenge_ids
        for g in dag.groups:
            ops = self.group_ops.get(g, [])
            seq: List[ProofTask] = []
            involved_at: Dict[str, int] = {}

            def involve(tids, idx):
                for t in tids:
                    involved_at[t] = idx

            for oid in ops:
                op = dag.ops[oid]
                out_bundle = list(op.outputs)
                ins = [i for i in op.inputs]
                beta, gamma = chal_of(g)
                if op.kind in ("matmul", "lmhead", "attn_score", "attn_pv"):
                    work = self._mm_work(op)
                    reads = [i for i in ins]
                    seq.append(ProofTask(f"sc:{oid[3:]}", g, "sc_matmul", reads, [gamma], work))
                    involve(reads + out_bundle, len(seq) - 1)
                elif op.kind in ("norm", "hadamard", "softmax_norm"):
                    work = sum(dag.tensors[o].elems for o in out_bundle[:1]) * 2
                    reads = [i for i in ins]
                    seq.append(ProofTask(f"sc:{oid[3:]}", g, "sc_hadamard", reads, [gamma], work))
                    involve(reads + out_bundle, len(seq) - 1)
                elif op.kind in ("residual", "argmax", "embed", "softmax_exp", "activation", "feed"):
                    # linear relations / lookup-only ops: claims reduce to openings
                    if seq:
                        involve(ins + out_bundle, len(seq) - 1)
                    else:
                        involve(ins + out_bundle, -1)
                for o in out_bundle:
                    lid = f"lk:{o[2:]}"
                    if lid in dag.lookups:
                        lk = dag.lookups[lid]
                        reads = lk.columns + [lk.helper]
                        work = dag.tensors[lk.helper].elems + lk.table
                        seq.append(ProofTask(f"lkp:{o[2:]}", g, "lookup", reads, [beta, gamma], work))
                        involve(reads, len(seq) - 1)
            # openings: one per committed tensor involved in this group, after its last use
            opens_after: Dict[int, List[str]] = {}
            for t, idx in involved_at.items():
                tt = dag.tensors[t]
                if tt.committed or tt.static:
                    opens_after.setdefault(idx, []).append(t)
            final: List[ProofTask] = []
            beta, gamma = chal_of(g)

            def emit_opens(idx):
                for t in sorted(opens_after.get(idx, []), key=lambda z: z):
                    tt = dag.tensors[t]
                    final.append(ProofTask(f"open:{t}@{g[0]}.{g[1]}", g, "open", [t], [gamma],
                                           tt.elems, target=t))

            emit_opens(-1)
            for i, task in enumerate(seq):
                final.append(task)
                emit_opens(i)
            for task in final:
                dag.tasks[task.id] = task

    def _mm_work(self, op: Op) -> int:
        if op.work:
            return op.work
        t = self.dag.tensors
        out = t[op.outputs[0]].elems
        ins = sum(t[i].elems for i in op.inputs)   # activation + weight (K*N) elements
        return out + ins

    def _challenge_ids(self, g):
        if self.enc.protocol == "GLOBAL":
            return "beta", "gamma"
        return f"beta@{g[0]}.{g[1]}", f"gamma@{g[0]}.{g[1]}"

    def _challenges_and_transcript(self):
        dag = self.dag
        tok_first = {}
        for t in dag.tensors.values():
            if t.role == "token" and t.producer is None:
                tok_first[t.group] = t.id
        helpers = [dag.lookups[l].helper for l in dag.lookups]
        if self.enc.protocol == "GLOBAL":
            order_base = []
            for g in dag.groups:
                if g in tok_first:
                    order_base.append(tok_first[g])
                for oid in self.group_ops.get(g, []):
                    for o in dag.ops[oid].outputs:
                        if dag.tensors[o].committed:
                            order_base.append(o)
            dag.challenges["beta"] = Challenge("beta", "beta", list(order_base))
            dag.challenges["gamma"] = Challenge("gamma", "gamma", list(order_base) + helpers)
            tr = [("commit", t) for t in order_base] + [("challenge", "beta")]
            tr += [("commit", h) for h in helpers] + [("challenge", "gamma")]
            tr += [("task", tid) for tid in dag.tasks]
            dag.transcript = tr
            return
        # LAYERWISE: per group, chained
        tr: List[Tuple[str, str]] = []
        committed_so_far: List[str] = []
        tasks_by_group: Dict[Tuple[int, int], List[str]] = {}
        for tid, t in dag.tasks.items():
            tasks_by_group.setdefault(t.group, []).append(tid)
        helpers_by_group: Dict[Tuple[int, int], List[str]] = {}
        for lk in dag.lookups.values():
            helpers_by_group.setdefault(lk.group, []).append(lk.helper)
        for g in dag.groups:
            gb = []
            if g in tok_first:
                gb.append(tok_first[g])
            for oid in self.group_ops.get(g, []):
                for o in dag.ops[oid].outputs:
                    if dag.tensors[o].committed:
                        gb.append(o)
            committed_so_far += gb
            tr += [("commit", t) for t in gb]
            b = f"beta@{g[0]}.{g[1]}"
                        # binds every commitment so far (transcript chaining): in particular all tensors this group's tasks read, including cross-group KV and checkpoints
            dag.challenges[b] = Challenge(b, "beta", list(committed_so_far), g)
            tr.append(("challenge", b))
            hs = helpers_by_group.get(g, [])
            committed_so_far += hs
            tr += [("commit", h) for h in hs]
            gm = f"gamma@{g[0]}.{g[1]}"
            dag.challenges[gm] = Challenge(gm, "gamma", list(committed_so_far), g)
            tr.append(("challenge", gm))
            tr += [("task", tid) for tid in tasks_by_group.get(g, [])]
        dag.transcript = tr

def build_dag(model: str | ModelShape, prefill: int, generated: int,
              enc: Optional[Encoding] = None, decode_block: int = 0) -> CertDAG:
    shape = model if isinstance(model, ModelShape) else get_shape(model)
    enc = enc or Encoding()
    dag = _Builder(shape, Workload(prefill, generated, decode_block), enc).build()
    for op in dag.derive_ops.values():
        op.challenge = "beta" if enc.protocol == "GLOBAL" else f"beta@{op.group[0]}.{op.group[1]}"
    return dag

def dag_example_json(dag: CertDAG, max_nodes: int = 400) -> dict:
    """Compact node/edge export (for the F2 legal proof/memory DAG figure)."""
    nodes = []
    for t in dag.tensors.values():
        nodes.append({"id": t.id, "type": "tensor", "role": t.role, "group": list(t.group),
                      "bytes": t.nbytes, "committed": t.committed, "anchor": t.anchor,
                      "static": t.static})
    for o in list(dag.ops.values()) + list(dag.derive_ops.values()):
        nodes.append({"id": o.id, "type": "op", "kind": o.kind, "group": list(o.group),
                      "macs": o.macs})
    for t in dag.tasks.values():
        nodes.append({"id": t.id, "type": "task", "kind": t.kind, "group": list(t.group),
                      "work": t.work})
    for c in dag.challenges.values():
        nodes.append({"id": c.id, "type": "challenge", "kind": c.kind})
    return {
        "summary": dag.summary(),
        "nodes": nodes[:max_nodes],
        "truncated": len(nodes) > max_nodes,
        "edges": {
            "inference": dag.inference_edges()[: max_nodes * 3],
            "proof": dag.proof_edges()[: max_nodes * 3],
            "commit_challenge": dag.commit_challenge_edges()[: max_nodes * 3],
        },
        "transcript_head": dag.transcript[:60],
    }
