"""Hash commitments with 256-bit random openings (PROTOTYPE ONLY).

    Com(m; r) = SHA-256( DST || u64le(|m|) || m || r ),  r <- {0,1}^256

Properties (stated, not proven here):
  * Binding: finding (m, r) != (m', r') with equal commitments yields a
    SHA-256 collision (computational binding under collision resistance).
  * Hiding: modelling SHA-256 as a random oracle, Com(m; r) with a fresh
    uniform 256-bit r is statistically close to uniform and independent of m
    (hiding in the random-oracle model).  SHA-256 is not a random oracle; this
    is a heuristic argument appropriate for a prototype.
  * A commitment reused with the same r for different messages, or with a
    low-entropy / predictable r, is NOT hiding.  Never derive r from a
    benchmark seed in production (see spec/leakage.md).

These commitments bind session inputs, context-state boundaries and private outputs in the EDGE-CERT public instance.  They are not polynomial commitments and are not what a real ZK backend uses for witness tables.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass

DST_COMMIT = b"EDGE-CERT/commit/v1"
OPENING_BYTES = 32

@dataclass(frozen=True)
class Opening:
    message: bytes
    randomness: bytes

def commit(message: bytes, randomness: bytes | None = None, dst: bytes = DST_COMMIT) -> tuple[bytes, Opening]:
    """Return (commitment, opening).  Fresh CSPRNG randomness unless supplied."""
    r = randomness if randomness is not None else secrets.token_bytes(OPENING_BYTES)
    if len(r) != OPENING_BYTES:
        raise ValueError("commitment randomness must be 32 bytes")
    c = hashlib.sha256(dst + len(message).to_bytes(8, "little") + message + r).digest()
    return c, Opening(message, r)

def verify_opening(commitment: bytes, opening: Opening, dst: bytes = DST_COMMIT) -> bool:
    if len(opening.randomness) != OPENING_BYTES:
        return False
    c = hashlib.sha256(dst + len(opening.message).to_bytes(8, "little")
                       + opening.message + opening.randomness).digest()
    return hmac.compare_digest(c, commitment)

def tagged_hash(tag: bytes, *parts: bytes) -> bytes:
    """Domain-separated, length-prefixed SHA-256 over several byte strings."""
    h = hashlib.sha256()
    h.update(len(tag).to_bytes(4, "little") + tag)
    for p in parts:
        h.update(len(p).to_bytes(8, "little") + p)
    return h.digest()
