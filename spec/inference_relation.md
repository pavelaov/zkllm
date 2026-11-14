# The EDGE-CERT inference relation R_EDGE (C1, v1)

## 1. Notation

`p = 2^64 - 2^32 + 1` (Goldilocks), `Ext = F_p[X]/(X^2 - 7)` (challenge field, `|Ext| = p^2`). `H(.)` is SHA-256; every hash use has a distinct domain-separation tag (Section 9). `canon(x)` is canonical JSON (sorted keys, `,`/`:` separators, UTF-8, no floats, integers `|v| <= 2^53 - 1`, digests as lower-case hex). `lift(v) = v mod p` embeds a signed integer. For a table `v` of length `2^k`, `v~` is its multilinear extension with LSB-first variable order (the order used by `edgecert.sumfold`).

## 2. Manifest (publisher authorization)

A signed manifest `M = (body, sig)` (`spec/manifest_schema.json`) binds: publisher key, model family/version, checkpoint digest, the model commitment `C_W` and commitment scheme, numeric tier, graph digest (and inline graph spec for the prototype), tokenizer, quantizer (per-layer multiplier `M_l`, shift `s_l`, rounding rule, clamp), lookup-table digests, positional encoding, mask semantics, decoding rule (greedy, lowest-index tie break, EOS id, generation cap), authorized adapters (each with its adapter commitment and combined model commitment), proof system (relation id, proof configuration id, backend, field, challenge field, hash, ZK/succinct flags, claimed security bits, verifier key id, setup), and policy (epoch, rollback rule, issue time, expiry, revoked manifest ids).

* `sig = Ed25519(sk_pub, "EDGE-CERT/manifest/v1\0" || canon(body))`
* `manifest_id = H("EDGE-CERT/manifest-id/v1\0" || canon(body))`
* `Auth(M) = {C_W} U {combined_model_commitment(a) : a in M.adapters}`, `combined(C_W, C_A) = H("EDGE-CERT/model+adapter/v1" || C_W || C_A)`.

Manifest acceptance (performed by the verifier, outside the arithmetic proof): `Accept(M)` iff the schema validates, the key id is in the verifier's trusted-publisher set and the embedded public key equals the trusted key, the signature verifies, `manifest_id` is not revoked, `M.policy_epoch >= floor(key, family)`, and `M` has not expired. On acceptance with `rollback_protection = monotonic-epoch` the floor is raised to `M.policy_epoch`.

## 3. Public instance

| Field | Type / encoding | Meaning |
|---|---|---|
| `manifest_id` | hex32 | Accepted manifest that authorizes this proof |
| `model_commitment` | hex32 | Must be in `Auth(M)` |
| `relation_id` | string | Must equal `M.proof_system.relation_id` (prototype: `R_EDGE/window-mlp-lm/v1`) |
| `proof_config_id` | string | Must equal `M.proof_system.proof_config_id` and be supported by the verifier |
| `session_id` | hex16 | Requester-chosen, fresh |
| `requester_nonce` | hex32 | Requester-chosen, fresh CSPRNG; single use at the verifier |
| `input_commitment` | hex32 | `Com_in = H("EDGE-CERT/input-commitment/v1" || u64(len) || canon({tokens, max_new_tokens, tokenizer_id}) || rho_in)`, `rho_in` 256-bit |
| `output_tokens_or_commitment` | `{mode:"tokens", tokens:[...]}` or `{mode:"commitment", commitment:hex32}` | Public output, or hiding commitment to it (`EDGE-CERT/output-commitment/v1`) |
| `context_anchor` | hex32 | `GENESIS_ANCHOR = H("EDGE-CERT/context-anchor/genesis/v1")` for a fresh session, else the previous receipt's end-state commitment |
| `output_length` | int | Number of generated tokens `T` (EOS included when produced) |
| `previous_receipt_id` | hex32 | `00..00` for a fresh session, else the id of a verifier-signed receipt |
| `policy_epoch` | int | Must equal `M.policy_epoch` and be `>=` the verifier's current floor |

`instance_digest = H("EDGE-CERT/instance/v1\0" || canon(I))`. Every challenge of every proof component is derived from a transcript whose initial state is `H("EDGE-CERT/fs/v1" || instance_digest)`.

## 4. Witness

`w = (W, r_W, adapter delta if any, LUTs, prompt, max_new_tokens, rho_in, state opening (window, position, rho_s) if chunked, end-state opening (window', position', rho_s'), output opening if committed, all intermediate integer tensors of Section 5, and backend randomness)`. In TRANSPARENT_REFERENCE mode the whole witness is revealed; in a ZK backend it is not.

## 5. Matrix-multiplication reduction (how C5 is proven)

For layer `l` with `ACC` of shape `out x T` padded to `2^a x 2^t` and inner dimension `K+1` padded to `2^b`: after all witness tables are committed and absorbed into the transcript, the verifier's challenges `r1 in Ext^a`, `r2 in Ext^t` define the claim `c = ACC~(r1, r2)`. The identity `ACC~(r1, r2) = sum_k Waug~(r1, k) * Y~(k, r2)` is proven with one product sumcheck (`edgecert.sumfold` semantics: `f[k] = Waug~(r1, k)`, `g[k] = Y~(k, r2)`, `b` rounds, degree 2), whose round transcript is the golden SumFold transcript with its initial state bound to the proof transcript state and the label `L<l>.matmul` (`crypto/transcript.py`). The final values must equal `Waug~(r1, r_k)` and `Y~(r_k, r2)`: in TRANSPARENT_REFERENCE mode the verifier evaluates these from the opened tables; in a succinct backend they are polynomial-commitment openings.

Transcript order (prover and verifier): `T_0 = H(fs/v1 || instance_digest)`; absorb end-state commitment; absorb the Merkle roots of `Y0` and of `ACC, Q, REM, SIG, OUT` for every layer; then per layer: squeeze `r1`, `r2`; run the bound sumcheck; absorb its final state and final values.

## 6. Instantiations

| | TRANSPARENT_REFERENCE (implemented) | ZK backend |
|---|---|---|
| C1 to C3, C9, C10 openings | Revealed values + Merkle / hash-commitment openings | Hiding commitments, opened inside the proof |
| C4, C6 (LUT), C8 | Direct integer checks on revealed tables | Lookup / range arguments of the backend |
| C5 | Product sumcheck per layer, final values from revealed tables | Same reduction, PCS openings |
| C7 | Direct canonical-range checks | Range arguments (limb lookups) |

## 7. Domain-separation tags

`EDGE-CERT/manifest/v1`, `EDGE-CERT/manifest-id/v1`, `EDGE-CERT/instance/v1`, `EDGE-CERT/fs/v1`, `EDGE-CERT/sumfold/v1` (golden), `EDGE-CERT/sumfold-bind/v1`, `EDGE-CERT/input-commitment/v1`, `EDGE-CERT/state-commitment/v1`, `EDGE-CERT/output-commitment/v1`, `EDGE-CERT/context-anchor/genesis/v1`, `EDGE-CERT/model-commitment/v1`, `EDGE-CERT/model+adapter/v1`, `EDGE-CERT/merkle/empty/v1`, `EDGE-CERT/graph/v1`, `EDGE-CERT/lut/v1`, `EDGE-CERT/tokenizer/v1`, `EDGE-CERT/receipt/v1`, `EDGE-CERT/receipt-id/v1`, `EDGE-CERT/audit/v1`, `EDGE-CERT/audit/genesis/v1`, `EDGE-CERT/audit-checkpoint/v1`, `EDGE-CERT/audit-cosign/v1`, `EDGE-CERT/key-id/v1`, `EDGE-CERT/commit/v1`.
