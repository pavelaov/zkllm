# Architecture

```
 model checkpoint (safetensors)      prompt / private request
            |                                  |
            v                                  v
  +------------------------+        +---------------------------+
  | signed manifest        |        | session context           |
  | (Ed25519, digest-bound)|        | (commitment, nonce, epoch) |
  +-----------+------------+        +-------------+-------------+
              |                                   |
              v                                   |
  +--------------------------+                    |
  | TILECONTRACT compiler    |   per-op contracts  |
  | graph -> gadgets ->      |--------------------+
  | ranges, accumulator      |                    |
  | bounds, lookups, witness |                    |
  +-----------+--------------+                    |
              |                                   |
              v                                   v
  +--------------------------+        +--------------------------+
  | CERTSCHED scheduler      |        | prover (device/host)     |
  | commit/challenge order,  |        | SUMFOLD product sumcheck |
  | device memory cap,       |        | CPU (C/OpenMP) | CUDA    |
  | spills, recompute        |        | | Zynq RTL (AXI-Lite)    |
  +-----------+--------------+        +-------------+------------+
              |  legal schedule                    |  transcript proof
              +------------------+-----------------+
                                 v
                    +---------------------------+
                    | honest verifier           |
                    | independent checks,       |
                    | receipts, session state   |
                    +---------------------------+
```

Module map (all paths under the artifact root):

| Module | Entry points |
| :--- | :--- |
| TILECONTRACT | `src/edgecert/compiler/graph.py`, `gadgets.py`, `tile_contract.py` (lower, static_check, account), `model_shapes.py` |
| CERTSCHED | `src/edgecert/runtime/dag.py`, `scheduler.py` (plan_certsched, check_schedule, run_policy), `runtime/simulate.py` |
| SUMFOLD | `src/edgecert/sumfold.py` (golden model), `kernels/cpu/`, `kernels/cuda/`, `fpga/rtl/`, `fpga/scripts/`, `fpga/host/` |
| signed manifests, sessions, verification | `src/edgecert/runtime/manifest.py`, `runtime/session.py`, `verifier/`, `crypto/`, `audit/` |
