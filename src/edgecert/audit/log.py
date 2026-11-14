"""Hash-linked receipt log with simulated external witnesses.

Model (spec/threat_model.md, audit):
  * The device operator runs an append-only log of receipt ids.  Entry i is
        entry_i = {"seq": i, "receipt_id": ..., "prev_head": head_{i-1}}
        head_i  = SHA-256("EDGE-CERT/audit/v1\\x00" || head_{i-1} || canonical(entry_i))
    with head_0 = SHA-256("EDGE-CERT/audit/genesis/v1" || log_id).
  * The operator signs checkpoints (log_id, seq, head).
  * External witnesses (independent parties, simulated in-process here)
    accept a new checkpoint only with a consistency proof (the entries since
    their last checkpoint), remember every head they verified, and cosign.
  * Relying parties verify a presented log against witnessed checkpoints.

Guarantee (conditional): if at least one honest witness verified checkpoint (seq, head) and the relying party consults it, then (a) a presented log shorter than seq is detected as ROLLBACK, (b) a presented log whose head at seq differs is detected as FORK, and (c) two operator-signed checkpoints with the same seq and different heads are non-repudiable EQUIVOCATION evidence once witnesses gossip.  Without an honest, consulted witness, a malicious operator can fork or roll back undetected; the prototype simulates witnesses and does not provide a real transparency service.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field as dfield
from typing import Dict, List, Mapping, Optional, Tuple

from ..crypto import ed25519
from ..runtime.manifest import canonical_json

AUDIT_DST = b"EDGE-CERT/audit/v1\x00"
CKPT_DST = b"EDGE-CERT/audit-checkpoint/v1\x00"
COSIG_DST = b"EDGE-CERT/audit-cosign/v1\x00"

def genesis_head(log_id: str) -> str:
    return hashlib.sha256(b"EDGE-CERT/audit/genesis/v1" + log_id.encode()).hexdigest()

def next_head(prev_head: str, entry: Dict) -> str:
    return hashlib.sha256(AUDIT_DST + bytes.fromhex(prev_head) + canonical_json(entry)).hexdigest()

def chain_heads(log_id: str, entries: List[Dict]) -> List[str]:
    """heads[k] = head after k entries (heads[0] = genesis).  Raises on broken links."""
    heads = [genesis_head(log_id)]
    for i, e in enumerate(entries):
        if e.get("seq") != i + 1 or e.get("prev_head") != heads[-1]:
            raise ValueError(f"broken link at entry {i + 1}")
        heads.append(next_head(heads[-1], e))
    return heads

def checkpoint_msg(log_id: str, seq: int, head: str) -> bytes:
    return CKPT_DST + canonical_json({"log_id": log_id, "seq": seq, "head": head})

class AuditLog:
    """Operator-side log.  A malicious operator may call ``fork``/``truncate``."""

    def __init__(self, log_id: str, operator_key: ed25519.SigningKey):
        self.log_id = log_id
        self.key = operator_key
        self.entries: List[Dict] = []

    def head(self) -> Tuple[int, str]:
        heads = chain_heads(self.log_id, self.entries)
        return len(self.entries), heads[-1]

    def append(self, receipt_id: str) -> Dict:
        seq, head = self.head()
        e = {"seq": seq + 1, "receipt_id": receipt_id, "prev_head": head}
        self.entries.append(e)
        return e

    def checkpoint(self) -> Dict:
        seq, head = self.head()
        sig = self.key.sign(checkpoint_msg(self.log_id, seq, head))
        return {"log_id": self.log_id, "seq": seq, "head": head, "operator_key_id": self.key.key_id,
                "sig": sig.hex()}

    # ---- misbehaviour helpers used by the attack suite
    def fork(self) -> "AuditLog":
        other = AuditLog(self.log_id, self.key)
        other.entries = [dict(e) for e in self.entries]
        return other

    def truncate(self, seq: int) -> None:
        self.entries = self.entries[:seq]

    def entries_between(self, lo: int, hi: int) -> List[Dict]:
        return [dict(e) for e in self.entries[lo:hi]]

@dataclass
class Witness:
    """Simulated external witness.  Keeps every head it has verified."""
    name: str
    key: ed25519.SigningKey
    operator_keys: Dict[str, bytes]
    heads: Dict[str, Dict[int, str]] = dfield(default_factory=dict)      # log_id -> seq -> head
    checkpoints: Dict[str, Dict[int, Dict]] = dfield(default_factory=dict)  # signed checkpoints seen
    alarms: List[Dict] = dfield(default_factory=list)

    def _alarm(self, kind: str, **info) -> Dict:
        a = {"witness": self.name, "kind": kind, **info}
        self.alarms.append(a)
        return a

    def observe(self, ckpt: Dict, entries_since_last: List[Dict]) -> Tuple[bool, Optional[Dict]]:
        """Verify and cosign a checkpoint.  Returns (accepted, cosignature | alarm)."""
        log_id, seq, head = ckpt["log_id"], ckpt["seq"], ckpt["head"]
        kid = ckpt["operator_key_id"]
        if kid not in self.operator_keys or not ed25519.verify(
                self.operator_keys[kid], checkpoint_msg(log_id, seq, head), bytes.fromhex(ckpt["sig"])):
            return False, self._alarm("BAD_CHECKPOINT_SIGNATURE", log_id=log_id, seq=seq)
        known = self.heads.setdefault(log_id, {0: genesis_head(log_id)})
        cps = self.checkpoints.setdefault(log_id, {})
        last = max(known)
        if seq in known and known[seq] != head:
            return False, self._alarm("EQUIVOCATION", log_id=log_id, seq=seq, seen=known[seq], presented=head,
                                      evidence=[cps.get(seq), ckpt])
        if seq < last:
            if known.get(seq) == head:
                return True, None  # old but consistent
            return False, self._alarm("ROLLBACK", log_id=log_id, seq=seq, last_seq=last)
        # consistency: entries (last, seq] must extend known[last] to head
        cur = known[last]
        if len(entries_since_last) != seq - last:
            return False, self._alarm("MISSING_CONSISTENCY_PROOF", log_id=log_id, seq=seq, last_seq=last)
        new_heads = {}
        for k, e in enumerate(entries_since_last):
            if e.get("seq") != last + k + 1 or e.get("prev_head") != cur:
                return False, self._alarm("INCONSISTENT_EXTENSION", log_id=log_id, seq=seq, last_seq=last)
            cur = next_head(cur, e)
            new_heads[last + k + 1] = cur
        if cur != head:
            return False, self._alarm("INCONSISTENT_EXTENSION", log_id=log_id, seq=seq, last_seq=last)
        known.update(new_heads)
        cps[seq] = ckpt
        cosig = self.key.sign(COSIG_DST + checkpoint_msg(log_id, seq, head))
        return True, {"witness": self.name, "witness_key_id": self.key.key_id, "log_id": log_id, "seq": seq,
                      "head": head, "cosig": cosig.hex()}

    def latest(self, log_id: str) -> Tuple[int, str]:
        known = self.heads.get(log_id, {0: genesis_head(log_id)})
        s = max(known)
        return s, known[s]

    def gossip(self, other: "Witness") -> List[Dict]:
        """Compare verified heads on common sequence numbers; report forks."""
        found = []
        for log_id, mine in self.heads.items():
            theirs = other.heads.get(log_id, {})
            for s in sorted(set(mine) & set(theirs)):
                if mine[s] != theirs[s]:
                    found.append(self._alarm("FORK", log_id=log_id, seq=s, head_a=mine[s], head_b=theirs[s],
                                             witnesses=[self.name, other.name]))
                    break
        return found

def verify_presented_log(log_id: str, entries: List[Dict], witnesses: List[Witness],
                         receipt_id: Optional[str] = None) -> Tuple[bool, str]:
    """Relying-party check of a log presented by the operator."""
    try:
        heads = chain_heads(log_id, entries)
    except ValueError as e:
        return False, f"audit.broken_chain: {e}"
    for w in witnesses:
        known = w.heads.get(log_id, {})
        for s, h in known.items():
            if s >= len(heads):
                return False, f"audit.rollback (witness {w.name} saw seq {s}, log has {len(entries)})"
            if heads[s] != h:
                return False, f"audit.fork (head mismatch at seq {s} vs witness {w.name})"
    for i in range(len(witnesses)):
        for j in range(i + 1, len(witnesses)):
            mi, mj = witnesses[i].heads.get(log_id, {}), witnesses[j].heads.get(log_id, {})
            for s in set(mi) & set(mj):
                if mi[s] != mj[s]:
                    return False, f"audit.fork (witnesses {witnesses[i].name}/{witnesses[j].name} disagree at seq {s})"
    if receipt_id is not None and not any(e["receipt_id"] == receipt_id for e in entries):
        return False, "audit.receipt_not_logged"
    return True, "ok"
