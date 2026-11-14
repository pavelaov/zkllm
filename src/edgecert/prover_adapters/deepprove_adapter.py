"""Adapters for REAL zero-knowledge backends. 
The integration plan below is the documented next step (planning/03 "select one real proof backend", G1 gate).

Candidates (planning/04_Source_Ledger.md):
  * DeepProve (Lagrange Labs) [R07 paper, R26 software]  PRIMARY candidate:
    sumcheck + lookup based, targets full generated-sequence verification.
    The ePrint abstract and the repository README describe DIFFERENT benchmark
    configurations; keep them apart.  Licence ("Lagrange License") must be
    reviewed before building, redistributing or publishing results.
  * zkGPT [R02 paper, R25 artifact]  CPU reference (GPT-2); the paper labels
    its BN254 configuration as about 100-bit security; never relabel as 128-bit.
  * zkLLM [R01]  GPU-oriented alternative (tlookup, zkAttn).

Integration plan (G1), each step producing a result record under results/raw/:
  1. Pin: record repository URL, commit hash, toolchain versions and licence
     text digest; review the licence; no build before approval if the licence
     restricts use.
  2. Build in isolation (no remote shell scripts piped to a shell); record the
     exact build command and log.  No CUDA toolkit is installed on this host
     (docs/PAPER_BRIEF.md Section 7), so any GPU build path is blocked until
     the author approves installing one.
  3. Reproduce ONE author-provided example proof and its verification
     unchanged; record prover time, verifier time, proof size, peak host
     memory, security/ZK mode flags as documented by the backend.
  4. Export the EDGE-CERT toy model (window-mlp-lm/v1) and then GPT-2 in the
     backend's supported model format; check operator coverage against the
     manifest graph and FAIL CLOSED on any unsupported operator.
  5. Instance binding: feed instance_digest(I) (verifier/instance.py) into the
     backend's public inputs / transcript seed.  If the backend exposes no
     public-input hook, wrap it: the EDGE-CERT verifier checks the backend
     proof AND that the backend's public inputs (model commitment, input and
     output encodings) equal the fields of I.  Record which variant is used.
  6. Model commitment: map manifest.model.model_commitment to the backend's
     own weight commitment (commitment_scheme = "backend-pcs"); the publisher
     signs the backend commitment, never a separate file hash.
  7. Numeric semantics: align the backend's quantizer / requant / lookup rules
     with spec/numerical_semantics.md bit-exactly; mismatches are relation
     changes that need a new relation_id, not tuning knobs.
  8. Session/runtime: plug the adapter into runtime/session.py unchanged
     (strict release), and rerun bench/attacks/run_attacks.py with the backend.
"""
from __future__ import annotations

from typing import List, Optional

from .base import BackendCapabilities, BackendNotAvailable, ProverBackend

STATUS = "RUN"

class _PlannedBackend(ProverBackend):
    CAPS: BackendCapabilities

    def capabilities(self) -> BackendCapabilities:
        return self.CAPS

    def _unavailable(self):
        raise BackendNotAvailable(f"{self.CAPS.name}: status {STATUS}; see the integration plan in "
                                  f"prover_adapters/deepprove_adapter.py")

    def infer(self, ctx, priv) -> List[int]:
        self._unavailable()

    def prove(self, ctx, priv, out_tokens, deadline=None) -> Optional[dict]:
        self._unavailable()

class DeepProveAdapter(_PlannedBackend):
    """Primary real-ZK candidate (full generated-sequence verification)."""
    CAPS = BackendCapabilities(
        name="deepprove", status=STATUS, certificate_kind="ZK_CERTIFICATE (planned)", zero_knowledge=True,
        succinct=True, full_sequence=True, operators=[], field=None, commitment=None, claimed_security_bits=None)

class ZkGPTAdapter(_PlannedBackend):
    """CPU reference backend (GPT-2)."""
    CAPS = BackendCapabilities(
        name="zkgpt", status=STATUS, certificate_kind="ZK_CERTIFICATE (planned)", zero_knowledge=True,
        succinct=True, full_sequence=False, operators=[], field="BN254 (as reported by the authors)",
        commitment=None, claimed_security_bits=None,
        notes="Authors report about 100-bit security for their configuration; not reproduced here.")

class ZkLLMAdapter(_PlannedBackend):
    """GPU-oriented alternative backend."""
    CAPS = BackendCapabilities(
        name="zkllm", status=STATUS, certificate_kind="ZK_CERTIFICATE (planned)", zero_knowledge=True,
        succinct=True, full_sequence=False, operators=[], field=None, commitment=None, claimed_security_bits=None,
        notes="Requires a CUDA toolkit, which is not installed on this host.")

ADAPTERS = {"deepprove": DeepProveAdapter, "zkgpt": ZkGPTAdapter, "zkllm": ZkLLMAdapter}
