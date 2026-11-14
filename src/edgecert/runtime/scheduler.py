"""CertSched (C3): legality checker, memory policies and the CertSched planner.

A *schedule* is a linear list of ``Event``s over the nodes of a ``CertDAG`` (see ``edgecert.runtime.dag``). ``check_schedule`` decides legality:

  (a) availability: every tensor a proof task (or helper derivation) reads is on
      the device and bound to its commitment: either resident since production,
      restored from a host copy or an *encrypted* spill, or deterministically
      recomputed from available inputs and then re-bound (REBIND) to the existing
      commitment;
  (b) immutability: no committed tensor is modified after commitment, never
      re-committed, and a recomputed copy must match the committed content;
  (c) challenge causality: each challenge is derived only after every commitment
      it depends on; no task/derivation uses a challenge before it exists; no
      commitment a challenge depends on is added after that challenge;
  (d) transcript determinism: the absorbed transcript (commits, challenges, proof
      tasks) equals the DAG's canonical order, independent of timing/placement;
  plus completeness: every proof obligation runs exactly once, every witness is
      committed.

Memory caps are reported separately (``fits``); a schedule can be legal and still exceed a cap, and an illegal schedule is rejected whatever its footprint.

Policies: KEEP_ALL, LAYER_STREAM_NAIVE (illegal on purpose), HOST_STAGE, SPILL, RECOMPUTE and CERTSCHED (per-tensor choice minimising added time under a device cap; greedy by bytes saved per added second, verified by the checker, with an exact brute-force optimum for small instances).
"""
from __future__ import annotations

import bisect
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .dag import CertDAG, Op

# ------------------------------------------------------------------ events
INPUT = "INPUT"
COMPUTE = "COMPUTE"
COMMIT = "COMMIT"
CHALLENGE = "CHALLENGE"
DERIVE = "DERIVE"
PROVE = "PROVE"
OPEN = "OPEN"
FREE = "FREE"
DROP = "DROP"
OFFLOAD_HOST = "OFFLOAD_HOST"
RESTORE_HOST = "RESTORE_HOST"
SPILL = "SPILL"
RESTORE_SPILL = "RESTORE_SPILL"
RECOMPUTE = "RECOMPUTE"
REBIND = "REBIND"
MODIFY = "MODIFY"

TRANSCRIPT_KINDS = (COMMIT, CHALLENGE, PROVE, OPEN)

@dataclass
class Event:
    kind: str
    target: str
    phase: str = ""
    encrypted: bool = True      # SPILL only: private witness must be encrypted at rest

    def __repr__(self) -> str:  # compact for test failure messages
        return f"{self.kind}({self.target})"

# -------------------------------------------------------------- violations
RULES = {
    "A_UNAVAILABLE": "a",            # needed tensor not on device (freed / never restored)
    "A_UNBOUND_RECOMPUTE": "a",      # recomputed copy used before REBIND to its commitment
    "A_NONDETERMINISTIC": "a",       # recompute of a non-deterministic op
    "A_UNENCRYPTED_SPILL": "a",      # private witness spilled without encryption
    "B_MODIFIED_AFTER_COMMIT": "b",
    "B_ADAPT_AFTER_CHALLENGE": "b",  # modification after a challenge bound to it was derived
    "B_RECOMMIT": "b",
    "B_REBIND_MISMATCH": "b",        # recomputed content differs from the committed content
    "C_CHALLENGE_BEFORE_COMMIT": "c",
    "C_COMMIT_AFTER_CHALLENGE": "c",
    "C_TASK_BEFORE_CHALLENGE": "c",
    "C_USE_BEFORE_COMMIT": "c",
    "D_TRANSCRIPT_ORDER": "d",
    "E_MISSING_OBLIGATION": "e",
    "E_DUPLICATE_TASK": "e",
    "E_MISSING_COMMIT": "e",
    "X_MALFORMED": "x",
}

@dataclass
class Violation:
    index: int
    rule: str
    target: str
    detail: str = ""

@dataclass
class CheckResult:
    legal: bool
    violations: List[Violation]
    rule_counts: Dict[str, int]
    peak_device: int
    peak_host: int
    peak_ssd: int
    peak_unified: int
    fits: Optional[bool]
    n_events: int
    event_counts: Dict[str, int]
    boundary_bytes: Dict[str, int] = field(default_factory=dict)

    @property
    def rules(self) -> List[str]:
        return sorted(self.rule_counts)

def check_schedule(dag: CertDAG, events: Sequence[Event], device_cap: Optional[int] = None,
                   host_cap: Optional[int] = None, ssd_cap: Optional[int] = None,
                   unified: bool = False, keep: int = 64) -> CheckResult:
    """Replay ``events`` against ``dag`` and decide legality (rules a-d + completeness)."""
    T = dag.tensors
    viol: List[Violation] = []
    counts: Counter = Counter()

    def bad(i, rule, tgt, detail=""):
        counts[rule] += 1
        if len(viol) < keep:
            viol.append(Violation(i, rule, tgt, detail))

    dev: Dict[str, object] = {}       # tid -> content version of the device copy
    dev_ok: Set[str] = set()          # device copy verified equal to the committed content
    host: Dict[str, Tuple[object, bool]] = {}
    ssd: Dict[str, Tuple[object, bool]] = {}
    committed: Dict[str, object] = {}
    derived: Set[str] = set()
    dependents: Dict[str, List[str]] = defaultdict(list)   # tensor -> challenges binding it
    for c in dag.challenges.values():
        for d in c.depends:
            dependents[d].append(c.id)
    freed_at: Dict[str, int] = {}
    transcript: List[Tuple[str, str]] = []
    done: Counter = Counter()
    fresh = [0]
    intern: Dict[tuple, int] = {}      # content identity -> small int version

    def vid(key: tuple) -> int:
        return intern.setdefault(key, len(intern))

    b_dev = b_host = b_ssd = 0
    for t in T.values():
        if t.static:   # model weights: committed in the signed manifest, device resident
            dev[t.id] = vid(("static", t.id))
            committed[t.id] = dev[t.id]
            dev_ok.add(t.id)
            b_dev += t.nbytes
    pk_dev, pk_host, pk_ssd, pk_uni = b_dev, 0, 0, b_dev
    boundary: Dict[str, int] = {}
    ev_counts: Counter = Counter()

    def put_dev(tid, ver, ok):
        nonlocal b_dev
        if tid not in dev:
            b_dev += T[tid].nbytes
        dev[tid] = ver
        if ok:
            dev_ok.add(tid)
        else:
            dev_ok.discard(tid)

    def drop_dev(tid, i):
        nonlocal b_dev
        if tid in dev and not T[tid].static:
            b_dev -= T[tid].nbytes
            del dev[tid]
            dev_ok.discard(tid)
            freed_at[tid] = i

    def need_dev(i, tid, bound: bool):
        if tid not in dev:
            where = f"freed at event {freed_at[tid]}" if tid in freed_at else "never resident"
            bad(i, "A_UNAVAILABLE", tid, where)
            return False
        if bound and T[tid].committed:
            if tid not in committed:
                bad(i, "C_USE_BEFORE_COMMIT", tid)
                return False
            if tid not in dev_ok:
                bad(i, "A_UNBOUND_RECOMPUTE", tid)
                return False
        return True

    for i, ev in enumerate(events):
        k, tgt = ev.kind, ev.target
        ev_counts[k] += 1
        if k == INPUT:
            if tgt not in T or T[tgt].producer is not None:
                bad(i, "X_MALFORMED", tgt, "INPUT of a produced tensor")
            else:
                put_dev(tgt, vid(("in", tgt)), False)
        elif k in (COMPUTE, RECOMPUTE, DERIVE):
            op: Optional[Op] = dag.ops.get(tgt) or dag.derive_ops.get(tgt)
            if op is None:
                bad(i, "X_MALFORMED", tgt, "unknown op")
                continue
            if k == RECOMPUTE and not op.deterministic:
                bad(i, "A_NONDETERMINISTIC", tgt)
            if op.challenge and op.challenge not in derived:
                bad(i, "C_TASK_BEFORE_CHALLENGE", tgt, op.challenge)
                        # the original helper derivation must read the committed columns; a helper *re*-computation may use transient copies because its output is re-bound
            bound_inputs = op.kind == "derive" and k != RECOMPUTE
            for u in op.inputs:
                need_dev(i, u, bound_inputs)
                        # output content is a deterministic function of the actual input contents; it is unknown only when an input is missing altogether
            present = all(u in dev for u in op.inputs)
            ver = vid((op.id,) + tuple(dev[u] for u in op.inputs)) if present else vid(("broken", i))
            for o in op.outputs:
                if k == RECOMPUTE and o in dev:
                    continue      # resident copy kept; the duplicate output is discarded
                if k != RECOMPUTE and o in committed and committed[o] != ver:
                    # re-running a producer over a committed tensor is a modification
                    bad(i, "B_MODIFIED_AFTER_COMMIT", o, "overwritten by " + k)
                put_dev(o, ver, False)
        elif k == COMMIT:
            if not need_dev(i, tgt, False):
                continue
            if tgt in committed:
                bad(i, "B_RECOMMIT", tgt)
                continue
            for c in dependents.get(tgt, ()):
                if c in derived:
                    bad(i, "C_COMMIT_AFTER_CHALLENGE", tgt, c)
            committed[tgt] = dev[tgt]
            dev_ok.add(tgt)
            transcript.append(("commit", tgt))
        elif k == CHALLENGE:
            c = dag.challenges.get(tgt)
            if c is None:
                bad(i, "X_MALFORMED", tgt, "unknown challenge")
                continue
            missing = [d for d in c.depends if d not in committed]
            if missing:
                bad(i, "C_CHALLENGE_BEFORE_COMMIT", tgt, f"{len(missing)} missing, e.g. {missing[0]}")
            if tgt in derived:
                bad(i, "X_MALFORMED", tgt, "challenge derived twice")
            derived.add(tgt)
            transcript.append(("challenge", tgt))
            if not boundary:
                boundary = {"device": b_dev, "host": b_host, "ssd": b_ssd}
        elif k in (PROVE, OPEN):
            task = dag.tasks.get(tgt)
            if task is None:
                bad(i, "X_MALFORMED", tgt, "unknown task")
                continue
            for c in task.challenges:
                if c not in derived:
                    bad(i, "C_TASK_BEFORE_CHALLENGE", tgt, c)
            for r in task.reads:
                need_dev(i, r, True)
            done[tgt] += 1
            transcript.append(("task", tgt))
        elif k == FREE:
            drop_dev(tgt, i)
        elif k == DROP:
            drop_dev(tgt, i)
            if tgt in host:
                b_host -= T[tgt].nbytes
                del host[tgt]
            if tgt in ssd:
                b_ssd -= T[tgt].nbytes
                del ssd[tgt]
        elif k == OFFLOAD_HOST:
            if need_dev(i, tgt, False):
                if tgt not in host:
                    b_host += T[tgt].nbytes
                host[tgt] = (dev[tgt], tgt in dev_ok)
                drop_dev(tgt, i)
        elif k == RESTORE_HOST:
            if tgt not in host:
                bad(i, "A_UNAVAILABLE", tgt, "no host copy")
            else:
                ver, ok = host[tgt]
                put_dev(tgt, ver, ok)
        elif k == SPILL:
            if not ev.encrypted:
                bad(i, "A_UNENCRYPTED_SPILL", tgt)
            if need_dev(i, tgt, False):
                if tgt not in ssd:
                    b_ssd += T[tgt].nbytes
                ssd[tgt] = (dev[tgt], tgt in dev_ok)
                drop_dev(tgt, i)
        elif k == RESTORE_SPILL:
            if tgt not in ssd:
                bad(i, "A_UNAVAILABLE", tgt, "no spill copy")
            else:
                ver, ok = ssd[tgt]
                put_dev(tgt, ver, ok)
        elif k == REBIND:
            if not need_dev(i, tgt, False):
                continue
            if tgt not in committed:
                bad(i, "X_MALFORMED", tgt, "REBIND of an uncommitted tensor")
            elif dev[tgt] == committed[tgt]:
                dev_ok.add(tgt)
            else:
                bad(i, "B_REBIND_MISMATCH", tgt)
        elif k == MODIFY:
            if need_dev(i, tgt, False):
                fresh[0] += 1
                dev[tgt] = vid(("mod", fresh[0]))
                dev_ok.discard(tgt)
                if tgt in committed:
                    bad(i, "B_MODIFIED_AFTER_COMMIT", tgt)
                    if any(c in derived for c in dependents.get(tgt, ())):
                        bad(i, "B_ADAPT_AFTER_CHALLENGE", tgt)
        else:
            bad(i, "X_MALFORMED", tgt, "unknown event kind " + k)
        if b_dev > pk_dev:
            pk_dev = b_dev
        if b_host > pk_host:
            pk_host = b_host
        if b_ssd > pk_ssd:
            pk_ssd = b_ssd
        if b_dev + b_host > pk_uni:
            pk_uni = b_dev + b_host

    # (d) transcript determinism
    canon = dag.transcript
    if transcript != canon:
        j = next((x for x in range(min(len(transcript), len(canon))) if transcript[x] != canon[x]),
                 min(len(transcript), len(canon)))
        got = transcript[j] if j < len(transcript) else None
        exp = canon[j] if j < len(canon) else None
        bad(-1, "D_TRANSCRIPT_ORDER", str(got), f"position {j}: expected {exp}")
    # completeness
    miss = [t for t in dag.tasks if done[t] == 0]
    if miss:
        bad(-1, "E_MISSING_OBLIGATION", miss[0], f"{len(miss)} proof tasks never run")
        counts["E_MISSING_OBLIGATION"] += len(miss) - 1
    dup = [t for t, n in done.items() if n > 1]
    for t in dup:
        bad(-1, "E_DUPLICATE_TASK", t)
    unc = [t for t in dag.committed_tensors() if t not in committed]
    if unc:
        bad(-1, "E_MISSING_COMMIT", unc[0], f"{len(unc)} witnesses never committed")

    peak_eff = pk_uni if unified else pk_dev
    fits = None
    if device_cap is not None or host_cap is not None or ssd_cap is not None:
        fits = ((device_cap is None or peak_eff <= device_cap)
                and (host_cap is None or unified or pk_host <= host_cap)
                and (ssd_cap is None or pk_ssd <= ssd_cap))
    return CheckResult(sum(counts.values()) == 0, viol, dict(counts), pk_dev, pk_host, pk_ssd,
                       pk_uni, fits, len(events), dict(ev_counts), boundary)

# ------------------------------------------------------------ action list
@dataclass
class Action:
    kind: str
    target: str
    needs: List[str]
    makes: List[str]
    block: tuple
    phase: str           # inf | derive | prove
    inference: bool      # an inference op (its inputs form the inference working set)

def action_list(dag: CertDAG) -> List[Action]:
    """Canonical protocol actions (without memory management) for the DAG's protocol."""
    T = dag.tensors
    ops_by_group: Dict[tuple, List[Op]] = defaultdict(list)
    for op in dag.ops.values():
        ops_by_group[op.group].append(op)
    lk_by_group: Dict[tuple, list] = defaultdict(list)
    for lk in dag.lookups.values():
        lk_by_group[lk.group].append(lk)
    tasks_by_group: Dict[tuple, list] = defaultdict(list)
    for t in dag.tasks.values():
        tasks_by_group[t.group].append(t)

    def nonstatic(ts):
        return [t for t in ts if not T[t].static]

    def inf_actions(g, block):
        acts = []
        for t in T.values():
            if t.group == g and t.role == "token" and t.producer is None:
                acts.append(Action(INPUT, t.id, [], [t.id], block, "inf", False))
                acts.append(Action(COMMIT, t.id, [t.id], [], block, "inf", False))
        for op in ops_by_group.get(g, []):
            acts.append(Action(COMPUTE, op.id, nonstatic(op.inputs), list(op.outputs), block, "inf", True))
            for o in op.outputs:
                if T[o].committed:
                    acts.append(Action(COMMIT, o, [o], [], block, "inf", False))
        return acts

    def derive_actions(lks, block):
        acts = []
        for lk in lks:
            d = dag.derive_ops["d:" + lk.id[3:]]
            acts.append(Action(DERIVE, d.id, nonstatic(d.inputs), [lk.helper], block, "derive", False))
            acts.append(Action(COMMIT, lk.helper, [lk.helper], [], block, "derive", False))
        return acts

    def task_actions(tasks, block):
        acts = []
        for t in tasks:
            reads = nonstatic(t.reads)
            reads.sort(key=lambda r: T[r].role == "helper")   # columns before helpers
            acts.append(Action(OPEN if t.kind == "open" else PROVE, t.id, reads, [], block, "prove", False))
        return acts

    acts: List[Action] = []
    if dag.enc.protocol == "GLOBAL":
        for g in dag.groups:
            acts += inf_actions(g, ("inf", g))
        acts.append(Action(CHALLENGE, "beta", [], [], ("beta",), "derive", False))
        for g in dag.groups:
            acts += derive_actions(lk_by_group.get(g, []), ("derive", g))
        acts.append(Action(CHALLENGE, "gamma", [], [], ("gamma",), "prove", False))
        for g in dag.groups:
            acts += task_actions(tasks_by_group.get(g, []), ("prove", g))
    else:
        for g in dag.groups:
            blk = ("grp", g)
            acts += inf_actions(g, blk)
            acts.append(Action(CHALLENGE, f"beta@{g[0]}.{g[1]}", [], [], blk, "derive", False))
            acts += derive_actions(lk_by_group.get(g, []), blk)
            acts.append(Action(CHALLENGE, f"gamma@{g[0]}.{g[1]}", [], [], blk, "prove", False))
            acts += task_actions(tasks_by_group.get(g, []), blk)
    return acts

# ---------------------------------------------------------------- policies
RETAIN, HOST, SPILLD, RECOMP, NAIVE = "RETAIN", "HOST", "SPILL", "RECOMPUTE", "NAIVE"
POLICIES = ["KEEP_ALL", "LAYER_STREAM_NAIVE", "HOST_STAGE", "SPILL", "RECOMPUTE", "CERTSCHED"]

def uniform_decisions(dag: CertDAG, policy: str) -> Dict[str, str]:
    dec = {}
    for t in dag.tensors.values():
        if t.static:
            continue
        if policy == "KEEP_ALL":
            dec[t.id] = RETAIN
        elif policy == "LAYER_STREAM_NAIVE":
            dec[t.id] = NAIVE
        elif policy == "HOST_STAGE":
            dec[t.id] = HOST
        elif policy == "SPILL":
            dec[t.id] = SPILLD
        elif policy == "RECOMPUTE":
            dec[t.id] = RETAIN if (t.anchor or dag.producer(t.id) is None) else RECOMP
        else:
            raise ValueError(policy)
    return dec

def _anchor_closure(dag: CertDAG) -> Dict[str, Set[str]]:
    """For every produced non-anchor tensor: the anchors its recomputation needs."""
    memo: Dict[str, frozenset] = {}
    T = dag.tensors

    def anchors(tid: str) -> frozenset:
        if tid in memo:
            return memo[tid]
        op = dag.producer(tid)
        acc: Set[str] = set()
        if op is not None:
            for u in op.inputs:
                tu = T[u]
                if tu.static:
                    continue
                if tu.anchor or dag.producer(u) is None:
                    acc.add(u)
                else:
                    acc |= anchors(u)
        memo[tid] = frozenset(acc)
        return memo[tid]

    return {t: set(anchors(t)) for t in T if not T[t].static}

def emit_schedule(dag: CertDAG, decisions: Dict[str, str], acts: Optional[List[Action]] = None,
                  encrypt_spill: bool = True) -> List[Event]:
    """Turn per-tensor decisions into a concrete event list (legal unless NAIVE)."""
    T = dag.tensors
    acts = acts if acts is not None else action_list(dag)
    dec = lambda t: decisions.get(t, RETAIN)  # noqa: E731

    needs_at: Dict[str, List[int]] = defaultdict(list)
    inf_needs: Dict[str, List[int]] = defaultdict(list)
    for i, a in enumerate(acts):
        for t in a.needs:
            needs_at[t].append(i)
            if a.inference:
                inf_needs[t].append(i)
    # anchors must stay recoverable while any recompute that needs them may happen
    virt: Dict[str, List[int]] = defaultdict(list)
    if any(v == RECOMP for v in decisions.values()):
        closure = _anchor_closure(dag)
        for t, d in decisions.items():
            if d != RECOMP:
                continue
            for i in needs_at.get(t, ()):
                if acts[i].phase != "inf":
                    for a in closure.get(t, ()):
                        virt[a].append(i)
    all_needs: Dict[str, List[int]] = {}
    for t in set(needs_at) | set(virt):
        all_needs[t] = sorted(set(needs_at.get(t, [])) | set(virt.get(t, [])))

    def next_after(lst, i):
        if not lst:
            return None
        j = bisect.bisect_right(lst, i)
        return lst[j] if j < len(lst) else None

    on_dev: Set[str] = set()
    on_host: Set[str] = set()
    on_ssd: Set[str] = set()
    unbound: Set[str] = set()
    committed: Set[str] = set()
    ev: List[Event] = []

    def emit(kind, tgt, phase):
        ev.append(Event(kind, tgt, phase, encrypted=encrypt_spill))

    def recompute(t, i, phase, keep: Set[str]):
        op = dag.producer(t)
        local = []
        for u in op.inputs:
            tu = T[u]
            if tu.static or u in on_dev:
                continue
            if u in on_host:
                emit(RESTORE_HOST, u, phase)
            elif u in on_ssd:
                emit(RESTORE_SPILL, u, phase)
            elif dag.producer(u) is not None and not tu.anchor and dec(u) != NAIVE:
                recompute(u, i, phase, keep)
            else:
                continue   # unrecoverable input: the checker will reject the RECOMPUTE
            on_dev.add(u)
            local.append(u)
        emit(RECOMPUTE, op.id, phase)
        for o in op.outputs:
            if o not in on_dev:
                on_dev.add(o)
                if o in committed:
                    unbound.add(o)
                if o != t and o not in keep:
                    emit(FREE, o, phase)
                    on_dev.discard(o)
                    unbound.discard(o)
        for u in local:
            if u not in keep:
                emit(FREE, u, phase)
                on_dev.discard(u)
                unbound.discard(u)

    def acquire(t, i, phase, keep):
        if T[t].static:
            return
        if t in on_dev:
            if t in unbound:
                emit(REBIND, t, phase)
                unbound.discard(t)
            return
        d = dec(t)
        if d == NAIVE:
            return
        if t in on_host:
            emit(RESTORE_HOST, t, phase)
        elif t in on_ssd:
            emit(RESTORE_SPILL, t, phase)
        elif dag.producer(t) is not None and not T[t].anchor:
            recompute(t, i, phase, keep)
            if t in unbound:
                emit(REBIND, t, phase)
                unbound.discard(t)
            return
        else:
            return
        on_dev.add(t)

    def release(t, i, a: Action):
        if T[t].static:
            return
        nxt = next_after(all_needs.get(t), i)
        if nxt is None:
            if t in on_dev or t in on_host or t in on_ssd:
                emit(DROP, t, a.phase)
            on_dev.discard(t)
            on_host.discard(t)
            on_ssd.discard(t)
            unbound.discard(t)
            return
        if t not in on_dev:
            return
        if next_after(inf_needs.get(t), i) is not None:
            return           # still part of the inference working set (e.g. KV cache)
        d = dec(t)
        if d == NAIVE:
            # streaming inference: commit the fresh witness, then drop it immediately
            if acts[nxt].kind == COMMIT and acts[nxt].block == a.block:
                return
            emit(FREE, t, a.phase)
            on_dev.discard(t)
            return
        if acts[nxt].block == a.block or d == RETAIN:
            return
        if d == HOST:
            if t in on_host:
                emit(FREE, t, a.phase)
            else:
                emit(OFFLOAD_HOST, t, a.phase)
                on_host.add(t)
        elif d == SPILLD:
            if t in on_ssd:
                emit(FREE, t, a.phase)
            else:
                emit(SPILL, t, a.phase)
                on_ssd.add(t)
        else:  # RECOMP
            emit(FREE, t, a.phase)
        on_dev.discard(t)
        unbound.discard(t)

    for i, a in enumerate(acts):
        keep = set(a.needs)
        for t in a.needs:
            acquire(t, i, a.phase, keep)
        emit(a.kind, a.target, a.phase)
        if a.kind == COMMIT:
            committed.add(a.target)
        for t in a.makes:
            on_dev.add(t)
        for t in list(dict.fromkeys(a.needs + a.makes)):
            release(t, i, a)
    return ev

# --------------------------------------------------------------- cost model
@dataclass
class DeviceProfile:
    """Throughput constants for one execution device (see configs/cost_model.yaml)."""
    name: str
    int8_ops_per_s: float          # effective int8 ops/s (2 ops per MAC)
    mem_Bps: float                 # effective device memory bandwidth
    sc_elems_per_s: float          # sumcheck work elements per second
    commit_Bps: float              # witness bytes committed per second
    hash_Bps: float                # digest throughput used by REBIND
    link_Bps: Optional[float]      # device<->host (PCIe); None for unified memory
    ssd_read_Bps: float
    ssd_write_Bps: float
    aes_Bps: float                 # AES-GCM for encrypted spill (host CPU)
    unified: bool = False
    c_matmul: float = 1.0
    c_hadamard: float = 1.0
    c_lookup: float = 2.0
    c_open: float = 1.0
    c_derive: float = 3.0
    evidence: Dict[str, str] = field(default_factory=dict)

    def spill_w_Bps(self) -> float:
        rates = [self.ssd_write_Bps, self.aes_Bps] + ([self.link_Bps] if self.link_Bps else [])
        return min(rates)

    def spill_r_Bps(self) -> float:
        rates = [self.ssd_read_Bps, self.aes_Bps] + ([self.link_Bps] if self.link_Bps else [])
        return min(rates)

    def host_Bps(self) -> float:
        return self.link_Bps if self.link_Bps else self.mem_Bps

CATEGORIES = ["inference", "commit", "derive", "sumcheck", "lookup", "open", "host_io",
              "spill_io", "recompute", "rebind"]

def op_time(dag: CertDAG, op: Op, prof: DeviceProfile, autoregressive: bool) -> float:
    T = dag.tensors
    if op.kind == "derive":
        return T[op.outputs[0]].elems * prof.c_derive / prof.sc_elems_per_s
    rows = 1
    if autoregressive and op.group[0] > 0:
        segs = dag.workload.segments()
        rows = segs[op.group[0]][1]
    w_bytes = sum(T[u].nbytes for u in op.inputs if T[u].static)
    out_bytes = sum(T[o].nbytes for o in op.outputs)
    return 2.0 * op.macs / prof.int8_ops_per_s + (w_bytes * rows + out_bytes) / prof.mem_Bps

def schedule_time(dag: CertDAG, events: Sequence[Event], prof: DeviceProfile) -> Dict[str, object]:
    """Serial single-device time of an event list, by category and phase."""
    T = dag.tensors
    by_cat = dict.fromkeys(CATEGORIES, 0.0)
    by_phase: Dict[str, float] = defaultdict(float)
    for e in events:
        k = e.kind
        dt = 0.0
        cat = None
        if k == COMPUTE:
            dt, cat = op_time(dag, dag.ops[e.target], prof, True), "inference"
        elif k == RECOMPUTE:
            op = dag.ops.get(e.target) or dag.derive_ops[e.target]
            dt, cat = op_time(dag, op, prof, False), "recompute"
        elif k == DERIVE:
            dt, cat = op_time(dag, dag.derive_ops[e.target], prof, False), "derive"
        elif k == COMMIT:
            dt, cat = T[e.target].nbytes / prof.commit_Bps, "commit"
        elif k in (PROVE, OPEN):
            task = dag.tasks[e.target]
            c = {"sc_matmul": prof.c_matmul, "sc_hadamard": prof.c_hadamard,
                 "lookup": prof.c_lookup, "open": prof.c_open}[task.kind]
            dt = task.work * c / prof.sc_elems_per_s
            cat = {"sc_matmul": "sumcheck", "sc_hadamard": "sumcheck", "lookup": "lookup",
                   "open": "open"}[task.kind]
        elif k in (OFFLOAD_HOST, RESTORE_HOST):
            dt, cat = T[e.target].nbytes / prof.host_Bps(), "host_io"
        elif k == SPILL:
            dt, cat = T[e.target].nbytes / prof.spill_w_Bps(), "spill_io"
        elif k == RESTORE_SPILL:
            dt, cat = T[e.target].nbytes / prof.spill_r_Bps(), "spill_io"
        elif k == REBIND:
            dt, cat = T[e.target].nbytes / prof.hash_Bps, "rebind"
        if cat:
            by_cat[cat] += dt
            by_phase[e.phase or "?"] += dt
    total = sum(by_cat.values())
    overhead = by_cat["host_io"] + by_cat["spill_io"] + by_cat["recompute"] + by_cat["rebind"]
    return {"total_s": total, "by_category_s": by_cat, "by_phase_s": dict(by_phase),
            "overhead_s": overhead}

# ------------------------------------------------------------- CertSched
@dataclass
class Item:
    tid: str
    nbytes: int
    costs: Dict[str, float]            # option -> added seconds

    def best(self, allowed: Optional[Set[str]] = None) -> Tuple[str, float]:
        opts = [(c, o) for o, c in self.costs.items() if allowed is None or o in allowed]
        c, o = min(opts)
        return o, c

def candidate_items(dag: CertDAG, prof: DeviceProfile, acts: Optional[List[Action]] = None,
                    allow_host: bool = True) -> List[Item]:
    """Knapsack items: tensors whose later uses fall in other blocks than their production."""
    T = dag.tensors
    acts = acts if acts is not None else action_list(dag)
    prod_block: Dict[str, tuple] = {}
    blocks: Dict[str, Set[tuple]] = defaultdict(set)
    last_inf: Dict[str, int] = {}
    for i, a in enumerate(acts):
        for t in a.makes:
            prod_block.setdefault(t, a.block)
        for t in a.needs:
            if a.inference:
                last_inf[t] = i
    for i, a in enumerate(acts):
        for t in a.needs:
            if i > last_inf.get(t, -1) and a.block != prod_block.get(t):
                blocks[t].add(a.block)
    # recompute chains: ops from anchors (memoised union of op sets)
    chain_memo: Dict[str, frozenset] = {}

    def chain(tid: str) -> frozenset:
        if tid in chain_memo:
            return chain_memo[tid]
        op = dag.producer(tid)
        s = {op.id}
        if op.kind != "derive":     # helpers: columns are co-resident for their lookup task
            for u in op.inputs:
                tu = T[u]
                if tu.static or tu.anchor or dag.producer(u) is None:
                    continue
                s |= chain(u)
        chain_memo[tid] = frozenset(s)
        return chain_memo[tid]

    op_cost: Dict[str, float] = {}

    def ocost(oid):
        if oid not in op_cost:
            op = dag.ops.get(oid) or dag.derive_ops[oid]
            op_cost[oid] = op_time(dag, op, prof, False)
        return op_cost[oid]

    items = []
    for t, bl in blocks.items():
        tt = T[t]
        if tt.static or not tt.committed and not tt.anchor:
            continue
        n = len(bl)
        if n == 0:
            continue
        b = tt.nbytes
        costs = {}
        if allow_host and not prof.unified:
            costs[HOST] = b / prof.host_Bps() * (1 + n)
        costs[SPILLD] = b / prof.spill_w_Bps() + n * b / prof.spill_r_Bps()
        if not tt.anchor and dag.producer(t) is not None:
            rc = sum(ocost(o) for o in chain(t))
            costs[RECOMP] = n * (rc + b / prof.hash_Bps)
        items.append(Item(t, b, costs))
    return items

def greedy_cover(items: Sequence[Item], need: int, host_avail: float = float("inf"),
                 ssd_avail: float = float("inf")) -> Tuple[Dict[str, str], float, int]:
    """Choose tensors to evict (and how) so that >= need bytes leave the device.

    Min-cost covering greedy (2-approximation when host/SSD capacities do not bind):
    scan items by bytes saved per added second; an item that would overshoot the
    residual on its own only closes a *candidate* solution (partial set + that
    item) and is skipped, smaller items are added. The cheapest candidate wins; a
    clean-up pass then re-retains evictions that are not needed.
    Returns (decisions for evicted items, added seconds, bytes saved).
    """
    if need <= 0 or not items:
        return {}, 0.0, 0

    def allowed_opts(it, host_used, ssd_used):
        return {o: v for o, v in it.costs.items()
                if not (o == HOST and host_used + it.nbytes > host_avail)
                and not (o == SPILLD and ssd_used + it.nbytes > ssd_avail)}

    order = sorted(range(len(items)), key=lambda i: -(items[i].nbytes / max(items[i].best()[1], 1e-15)))
    picks: List[Tuple[int, str, float]] = []
    saved = 0
    cost = 0.0
    host_used = ssd_used = 0.0
    best_sol: Optional[Tuple[float, int, int, str, float]] = None   # cost, n_picks, j, option, c_j
    for idx in order:
        it = items[idx]
        r = need - saved
        if r <= 0:
            break
        allowed = allowed_opts(it, host_used, ssd_used)
        if not allowed:
            continue
        o = min(allowed, key=allowed.get)
        c = allowed[o]
        if it.nbytes >= r:
            if best_sol is None or cost + c < best_sol[0]:
                best_sol = (cost + c, len(picks), idx, o, c)
            continue
        picks.append((idx, o, c))
        saved += it.nbytes
        cost += c
        if o == HOST:
            host_used += it.nbytes
        elif o == SPILLD:
            ssd_used += it.nbytes
    if saved >= need and (best_sol is None or cost <= best_sol[0]):
        sel = picks
    elif best_sol is not None:
        sel = picks[:best_sol[1]] + [(best_sol[2], best_sol[3], best_sol[4])]
    else:
        sel = picks       # infeasible: everything evictable was evicted
    chosen: Dict[str, Tuple[str, float, int]] = {
        items[i].tid: (o, c, items[i].nbytes) for i, o, c in sel}
    saved_i = sum(v[2] for v in chosen.values())
    for tid, (o, c, b) in sorted(chosen.items(), key=lambda kv: -kv[1][1]):
        if saved_i - b >= need:
            del chosen[tid]
            saved_i -= b
    return ({t: v[0] for t, v in chosen.items()}, sum(v[1] for v in chosen.values()), saved_i)

def exact_cover(items: Sequence[Item], need: int, max_items: int = 22) -> Tuple[Optional[float], Optional[int]]:
    """Exact minimum added time to evict >= need bytes (brute force, small n).

    Uses each item's cheapest option (options are independent per item when the
    host and SSD capacities are not binding). Returns (min_cost, mask).
    """
    import numpy as np
    n = len(items)
    if n > max_items:
        raise ValueError("instance too large for brute force")
    if need <= 0:
        return 0.0, 0
    b = np.array([it.nbytes for it in items], dtype=np.float64)
    c = np.array([it.best()[1] for it in items], dtype=np.float64)
    best_cost, best_mask = None, None
    chunk = 1 << min(n, 18)
    total = 1 << n
    bits = np.arange(n, dtype=np.int64)
    for start in range(0, total, chunk):
        masks = np.arange(start, min(start + chunk, total), dtype=np.int64)
        sel = ((masks[:, None] >> bits[None, :]) & 1).astype(np.float64)
        sv = sel @ b
        cs = sel @ c
        ok = sv >= need
        if ok.any():
            j = int(np.argmin(np.where(ok, cs, np.inf)))
            if best_cost is None or cs[j] < best_cost:
                best_cost, best_mask = float(cs[j]), int(masks[j])
    return best_cost, best_mask

@dataclass
class PlanResult:
    policy: str
    decisions: Dict[str, str]
    events: List[Event]
    check: CheckResult
    time: Dict[str, object]
    iterations: int = 1
    feasible: Optional[bool] = None
    predicted_added_s: Optional[float] = None
    evicted_bytes: int = 0
    selected: str = "greedy"          # greedy | keep_all | uniform:<POLICY>
    alternatives: Dict[str, float] = field(default_factory=dict)

def block_working_set(dag: CertDAG, acts: List[Action]) -> int:
    T = dag.tensors
    ws: Dict[tuple, Set[str]] = defaultdict(set)
    for a in acts:
        if a.phase != "inf":
            ws[a.block].update(t for t in a.needs if not T[t].static)
    return max((sum(T[t].nbytes for t in s) for s in ws.values()), default=0)

def run_policy(dag: CertDAG, policy: str, prof: DeviceProfile, device_cap: Optional[int] = None,
               host_cap: Optional[int] = None, ssd_cap: Optional[int] = None,
               acts: Optional[List[Action]] = None) -> PlanResult:
    acts = acts if acts is not None else action_list(dag)
    if policy == "CERTSCHED":
        return plan_certsched(dag, prof, device_cap, host_cap, ssd_cap, acts=acts)
    dec = uniform_decisions(dag, policy)
    ev = emit_schedule(dag, dec, acts)
    chk = check_schedule(dag, ev, device_cap, host_cap, ssd_cap, prof.unified)
    return PlanResult(policy, dec, ev, chk, schedule_time(dag, ev, prof), feasible=chk.fits)

def plan_certsched(dag: CertDAG, prof: DeviceProfile, device_cap: Optional[int],
                   host_cap: Optional[int] = None, ssd_cap: Optional[int] = None,
                   acts: Optional[List[Action]] = None, max_iter: int = 8,
                   portfolio: bool = True) -> PlanResult:
    """Profile-guided per-tensor retain/host/spill/recompute choice under a device cap.

    The greedy plan is refined against the checker's replayed peak. With
    ``portfolio`` the uniform policies (special cases of per-tensor choices) are
    also replayed and the fastest legal schedule within all caps is returned;
    ``selected`` records which one won.
    """
    acts = acts if acts is not None else action_list(dag)
    res = _plan_greedy(dag, prof, device_cap, host_cap, ssd_cap, acts, max_iter)
    if not portfolio or res.selected == "keep_all":
        return res
    res.alternatives["greedy"] = res.time["total_s"] if res.check.fits and res.check.legal else float("inf")
    best = res
    for pol in (["SPILL", "RECOMPUTE"] if prof.unified else ["HOST_STAGE", "SPILL", "RECOMPUTE"]):
        alt = run_policy(dag, pol, prof, device_cap, host_cap, ssd_cap, acts=acts)
        ok = alt.check.legal and bool(alt.check.fits)
        res.alternatives[pol] = alt.time["total_s"] if ok else float("inf")
        if ok and (not (best.check.fits and best.check.legal)
                   or alt.time["total_s"] < best.time["total_s"]):
            alt.policy = "CERTSCHED"
            alt.selected = "uniform:" + pol
            alt.feasible = True
            alt.iterations = res.iterations
            best = alt
    best.alternatives = res.alternatives
    return best

def _plan_greedy(dag: CertDAG, prof: DeviceProfile, device_cap: Optional[int],
                 host_cap: Optional[int], ssd_cap: Optional[int], acts: List[Action],
                 max_iter: int) -> PlanResult:
    base = uniform_decisions(dag, "KEEP_ALL")
    ev0 = emit_schedule(dag, base, acts)
    chk0 = check_schedule(dag, ev0, device_cap, host_cap, ssd_cap, prof.unified)
    peak0 = chk0.peak_unified if prof.unified else chk0.peak_device
    if device_cap is None or peak0 <= device_cap:
        return PlanResult("CERTSCHED", base, ev0, chk0, schedule_time(dag, ev0, prof),
                          feasible=chk0.fits, predicted_added_s=0.0, selected="keep_all")
    items = candidate_items(dag, prof, acts, allow_host=not prof.unified)
    host_avail = float("inf") if host_cap is None or prof.unified else float(host_cap)
    ssd_avail = float("inf") if ssd_cap is None else float(ssd_cap)
    need = peak0 - device_cap
    best = None
    for it_no in range(1, max_iter + 1):
        chosen, added, saved = greedy_cover(items, need, host_avail, ssd_avail)
        dec = dict(base)
        dec.update(chosen)
        ev = emit_schedule(dag, dec, acts)
        chk = check_schedule(dag, ev, device_cap, host_cap, ssd_cap, prof.unified)
        peak = chk.peak_unified if prof.unified else chk.peak_device
        res = PlanResult("CERTSCHED", dec, ev, chk, schedule_time(dag, ev, prof), it_no,
                         chk.fits, added, saved)
        if best is None or peak < (best.check.peak_unified if prof.unified else best.check.peak_device):
            best = res
        if peak <= device_cap:
            return res
        if saved < need:     # every candidate already evicted
            break
                # the replayed peak can sit at a different point in time than the boundary the knapsack models; grow the eviction target geometrically until it fits
        need = max(need, saved) + max(peak - device_cap, 0.002 * device_cap) * (2 ** (it_no - 1))
    best.feasible = False
    return best
