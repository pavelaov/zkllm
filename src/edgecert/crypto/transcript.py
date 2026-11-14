"""Instance-bound Fiat-Shamir transcripts (shared by prover and verifier).

Two layers:

1. ``ProofTranscript``: one SHA-256 chain per proof,
       T_0 = SHA-256("EDGE-CERT/fs/v1" || instance_digest)
   into which the prover absorbs (in a fixed order) every commitment that a
   later challenge must depend on, and from which it squeezes extension-field
   challenges.  Commit-before-challenge is enforced by construction: a
   challenge is a hash of everything absorbed before it.

2. ``BoundSumcheckTranscript``: the SumFold round transcript of
   spec/sumfold_kernel.md Section 3 (identical ``round`` function, so the
   accelerator semantics are unchanged), whose initial state is additionally
   bound to the proof transcript state and a label:
       state_0' = SHA-256("EDGE-CERT/sumfold-bind/v1" || state_0 || ctx || label)
   where state_0 = SHA-256("EDGE-CERT/sumfold/v1" || u32le(n) || enc(H)) is the
   golden-model initial state.  Every SumFold challenge therefore depends on
   the public-instance digest and on all previously absorbed commitments.

The golden model (edgecert.sumfold) is imported, not modified.
"""
from __future__ import annotations

import hashlib
from typing import List

from ..field import P, Ext, enc
from ..sumfold import Transcript as GoldenTranscript

FS_DST = b"EDGE-CERT/fs/v1"
BIND_DST = b"EDGE-CERT/sumfold-bind/v1"

class ProofTranscript:
    def __init__(self, instance_digest: bytes):
        if len(instance_digest) != 32:
            raise ValueError("instance digest must be 32 bytes")
        self.state = hashlib.sha256(FS_DST + instance_digest).digest()
        self.log: List[str] = []  # labels only (for debugging / order audits)

    def absorb(self, label: str, data: bytes) -> None:
        lb = label.encode()
        self.state = hashlib.sha256(self.state + b"A" + len(lb).to_bytes(4, "little") + lb
                                    + len(data).to_bytes(8, "little") + data).digest()
        self.log.append("absorb:" + label)

    def absorb_ext(self, label: str, e: Ext) -> None:
        self.absorb(label, enc(e))

    def challenge_ext(self, label: str) -> Ext:
        lb = label.encode()
        self.state = hashlib.sha256(self.state + b"C" + len(lb).to_bytes(4, "little") + lb).digest()
        self.log.append("challenge:" + label)
        c0 = int.from_bytes(self.state[0:8], "little") % P
        c1 = int.from_bytes(self.state[8:16], "little") % P
        return (c0, c1)

    def challenge_vec(self, label: str, k: int) -> List[Ext]:
        return [self.challenge_ext(f"{label}[{i}]") for i in range(k)]

class BoundSumcheckTranscript(GoldenTranscript):
    """Golden SumFold transcript with its initial state bound to a context."""

    def __init__(self, n: int, claim: Ext, context: bytes, label: str):
        super().__init__(n, claim)
        lb = label.encode()
        self.state = hashlib.sha256(BIND_DST + self.state + len(context).to_bytes(4, "little") + context
                                    + len(lb).to_bytes(4, "little") + lb).digest()
