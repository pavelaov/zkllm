# Interfaces

The components talk through four documented interfaces; the authoritative definitions are `spec/manifest_schema.json` and `spec/sumfold_kernel.md`.

## 1. Signed model manifest (publisher -> device)

`runtime/manifest.py`. Envelope = `body` + `signature` over the canonical JSON of `body`.

- `manifest_version`: `edgecert-manifest/v1`
- `publisher`: `{name, key_id, alg: "Ed25519"}` plus `public_key` in the trust policy
- `model`: digest-bound graph description and quantisation policy (see schema)
- `policy`: commit/challenge protocol, tile sizes, rounding mode, table digests
- `validity`: `{issued_at, not_after}`, optional `revokes`
- Device-side acceptance: signature check, digest binding, epoch freshness, revocation list (`TrustPolicy`, `authorize_manifest`); rejections name the failing check (e.g. `manifest.bad_signature`, `manifest.rollback_stale_epoch`).

## 2. Session and certificate (requester <-> verifier)

`runtime/session.py`, `verifier/instance.py`, `audit/receipts.py`.

- `SessionContext`: `{manifest_id, session_id, requester_nonce, input_commitment, parent_receipt_id, anchor, epoch, output_mode}`; the nonce is single-use (`session.nonce_fresh`).
- Certificate: `{label: "TRANSPARENT_REFERENCE", zero_knowledge: false, manifest_id, session, input, output, sumchecks[], witness{commitments}, receipts[], challenges}`.
- Verification order: manifest authorization -> session/nonce -> input commitment -> per-op sumcheck replay -> accumulator commitment -> table digests -> receipt chain; the first failing check is reported.

## 3. Python API (compiler / scheduler / prover)

- `compiler.tile_contract.lower(ops, impl="gpu_imma_s8", field_name="goldilocks", rc_bits=16)` returns `{op_id: TileContract}`; `assert_static(contracts, p)` raises `ContractViolation`; `account(ops, workload, tables)` returns totals (lookups, witness bytes, tiles).
- `runtime.scheduler.plan_certsched(dag, profile, cap)` returns a plan with `.check.legal`, `.check.fits`, `.check.peak_device`, `.time["overhead_s"]`; `check_schedule(dag, events)` validates a hand-made schedule.
- `prover_adapters.transparent.TransparentProver(model, manifest, backend=...)` exposes `infer(ctx, request)` and `prove(ctx, request, output)`; backends: `HostSumcheck`, `SumFoldLaneEmulator(lanes)`, `TransparentProver` adapters (`deepprove_adapter`).

## 4. Kernel and FPGA ABI

- Kernel CLI (`kernels/cpu/main.c`, CUDA equivalent): `--logn <n> --threads <t> --variant {fused,unfused} --seedf <hex> --seedg <hex> --reps <r> --warmup <w> --sched {static,dynamic}` prints one JSON line per configuration (`total_s`, `pass_kinds`, transcript hash).
- Python bindings: `kernels/cpu/mle.c` and `kernels/cuda/shim/shim_launch.cpp` are loaded via `ctypes` by `bench/runners/kernel_common.py`.
- FPGA: `fpga/rtl/sumfold_axil.v` maps the AXI-Lite registers defined in `fpga/rtl/sumfold_defs.vh`; `fpga/scripts/build.tcl` takes `<lanes> <clock MHz>`; the board hosts (`fpga/host/board_probe.py`, `board_campaign.py`) speak the same JSON transcript format as the kernel CLI, so `fpga/sim/replay/replay.py` can replay either side.

## Data formats

Run records are JSON with `evidence_class`, `command`, `timestamp_utc`, `host` and the payload; golden vectors are little-endian hex (`fpga/sim/golden/*.hex`) indexed by case name as listed in `fpga/sim/golden/manifest.json`. Prompt sets are JSON arrays of int32 token ids (`data/prompts/`), and the perplexity passage is plain UTF-8 text.
