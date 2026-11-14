"""Backend interface shared by every EDGE-CERT prover adapter.

A backend turns (manifest, public session context, private request, provisional output) into a proof object for R_EDGE, and tells the runtime which parts of the relation it actually covers.  The runtime and verifier never trust a backend's self-description for security: coverage flags only drive fail-closed decisions (an adapter that does not cover an operator used by the manifest graph must not be selected, spec/security_obligations.md O3).
"""
from __future__ import annotations

from dataclasses import dataclass, field as dfield
from typing import List, Optional

class BackendNotAvailable(RuntimeError):
    """Raised by adapters whose backend has not been built / reproduced here (status NOT_RUN)."""

@dataclass
class BackendCapabilities:
    name: str
    status: str                      # "RUNNING_PROTOTYPE" | "NOT_RUN" | ...
    certificate_kind: str            # "TRANSPARENT_REFERENCE" | "ZK_CERTIFICATE" | ...
    zero_knowledge: bool
    succinct: bool
    full_sequence: bool              # proves every generated token incl. EOS/length
    operators: List[str] = dfield(default_factory=list)
    field: Optional[str] = None
    commitment: Optional[str] = None
    claimed_security_bits: Optional[int] = None   # as documented by the backend authors; never inferred here
    notes: str = ""

class ProverBackend:
    """Interface: ``infer`` returns provisional tokens, ``prove`` returns a proof or None."""

    def capabilities(self) -> BackendCapabilities:  # pragma: no cover - interface
        raise NotImplementedError

    def infer(self, ctx, priv) -> List[int]:  # pragma: no cover - interface
        raise NotImplementedError

    def prove(self, ctx, priv, out_tokens, deadline=None) -> Optional[dict]:  # pragma: no cover - interface
        raise NotImplementedError

TRANSPARENT_CAPABILITIES = BackendCapabilities(
    name="edgecert-transparent-reference", status="RUNNING_PROTOTYPE",
    certificate_kind="TRANSPARENT_REFERENCE", zero_knowledge=False, succinct=False, full_sequence=True,
    operators=["gather", "matmul+bias", "requant(HALF_UP)", "clamp", "lut", "argmax(LOWEST_INDEX)", "eos/length"],
    field="Goldilocks p=2^64-2^32+1 with Fp[X]/(X^2-7) challenges", commitment="SHA-256 Merkle (witness revealed)",
    claimed_security_bits=None,
    notes="Reference for the certification contract only; reveals the witness; not an inference certificate.")
