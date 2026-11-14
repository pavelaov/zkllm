# Artifact evaluation guide

The artifact is anonymous and code-only. It reproduces the paper's claims that are measurable on a CPU-only x86-64 machine (and on a single NVIDIA GPU if available). The FPGA flow is reproducible only with Vivado 2022.2 and a ZCU102 board, which are not part of the artifact; the RTL, the scripts and the golden vectors are included and the expected reports are described below.

## Environment

| Item | Version used for the paper | Comment |
| :--- | :--- | :--- |
| OS | Ubuntu 22.04 / Windows 11 | any x86-64 Linux or Windows works for the Python parts |
| Python | 3.10 - 3.13 | `numpy`, `pyyaml` only; `cupy` optional for GPU runs |
| C toolchain | gcc 11.4 / MSVC 19 | OpenMP for the multi-threaded CPU kernel |
| GPU | RTX 6000 Ada (48 GB) | any CUDA 8.9 GPU; CPU path is available for every claim |
| FPGA | Vivado 2022.2, ZCU102 (XCZU9EG) | required only for the FPGA claims |

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD/src          # Windows: set PYTHONPATH=%CD%\src
python scripts/get_models.py        # optional: GPT-2 124M and Qwen3-0.6B, SHA-256 verified
```

`get_models.py` downloads the two public checkpoints into `models/` and verifies `models/SHA256SUMS.txt`. No weights ship with the artifact; the licence of each checkpoint applies.

## Claims, modules and commands

| Paper claim | Module | Command (from the artifact root) | Output |
| :--- | :--- | :--- | :--- |
| Signed manifests, sessions, certificates accepted; tampering rejected | signed manifest contracts, verifier | `python demo/demo.py` (steps 1 and 4); `python -m unittest tests.test_security_manifest tests.test_security_transparent` | prints per-check verdicts; unit tests |
| Contract statistics (Tiles, Acc, GL/BB headroom, K-split, lookups, witness MB) for GPT-2, Qwen3, Gemma 3, Llama-3 | TILECONTRACT | `python bench/runners/contracts_eval.py` | `results/raw/contracts/*.json` |
| Legal schedules, optimality gap, debt/latency, load sweep | CERTSCHED | `python bench/runners/sched_cost_model.py`; `sched_static.py --jobs 1`; `sched_gap.py --instances 20`; `sched_des.py --seeds 2 --jobs 1`; `sched_summary.py` | `results/raw/sched/*.json`, `summary.json` |
| SumFold kernel timings on CPU and GPU | SUMFOLD | `python bench/runners/kernel_cpu_bench.py --compilers gcc --rebuild`; `python bench/runners/kernel_gpu_bench.py --devices 0 --warmup 1000 --skip-emulation` | `results/raw/kernels/cpu_bench.json`, `gpu_bench.json` |
| Transparent reference prover cost and proof sizes | SUMFOLD + verifier | `python bench/attacks/time_transparent.py --reps 5 --warmup 1` | `results/raw/attacks/transparent_scaling.json` |
| Attack/enforcement matrix (41 cases) | verifier + runtime | `python bench/attacks/run_attacks.py --reps 10 --warmup 3` | `results/raw/attacks/matrix.json` (suite wall, per-row verdicts) |
| RQ1 quality: top-1 agreement, perplexity, commits | TILECONTRACT + reference model | `python bench/runners/realmodel_eval.py --model gpt2 --backend cpu --out-suffix _local`; then `realmodel_finalize.py`; `realmodel_summary.py` (repeat with `--model qwen3`) | `results/raw/realmodels/*.json` |
| RQ2 latency and peak memory | CERTSCHED simulator | included in `sched_*.py` records | `results/raw/sched/*.json` |
| RQ4 GPT-2 linear-layer prover on CPU and GPU | signed manifests + SUMFOLD | `python bench/runners/gpt2_linear_prove.py --final` | `results/raw/gpt2prove/summary_final.json` |
| FPGA post-route results and board campaign | SUMFOLD RTL | `python fpga/scripts/launch_impl.py` (Vivado 2022.2), `collect_impl.py`, `fpga/host/board_probe.py`, `board_campaign.py`, `python fpga/sim/replay/replay.py` | implementation reports and board transcripts (requires hardware) |
| Tables and numbers of the paper | analysis | `python bench/analysis/make_tables.py`; `make_numbers.py`; `make_all.py` | `paper/tables/*.tex`, `paper/generated/numbers.tex` (needs the paper skeleton) |

## Verification-side checks

`tests/` contains the independent checks used during the review: contract lowering and negative cases, manifest signing/authorization, schedule legality, kernel self-tests, and the transparent proof accept/reject cases (a tampered sumcheck message and a tampered accumulator must be rejected, and a replayed proof must be refused by the consumed nonce). The same checks are exposed by `demo/demo.py`.

## FPGA flow (no binaries bundled)

```bash
vivado -mode batch -source fpga/scripts/build.tcl -tclargs L8 250     # synthesis + implementation
python fpga/scripts/launch_impl.py --configs L1_250MHz_v2,L2_250MHz_v2,L4_250MHz_v2,L8_250MHz_v2
python fpga/scripts/collect_impl.py                                   # parses the Vivado reports
python fpga/host/board_probe.py --bitstream fpga/build/L8_250MHz_v2/sumfold_top.bit
python fpga/host/board_campaign.py --jobs 36                          # ZCU102 over JTAG-to-AXI
```

The RTL entry point is `fpga/rtl/sumfold_top.v` (AXI-Lite wrapper: `sumfold_axil.v`), the constraints are `fpga/constraints/zcu102.xdc`, and the golden vectors for RTL simulation are in `fpga/sim/golden/` (`manifest.json` lists every case). Expected post-route utilisation, timing and power for the five configurations are in `fpga/reports/*/summary.txt`; the full Vivado reports are regenerated by the flow above.


## Full reproducibility record from the paper

This section preserves the complete text of the paper's original Appendix D, **Artifact and Reproducibility**, with Markdown formatting. It records the original experimental environment and reported results; this migration does not report a new execution. The pre-existing environment table above lists different Python and compiler versions and has been retained unchanged. Paths and generated records below describe the full experimental repository; the minimal code-only release does not bundle the `results/` records or the `paper/` skeleton. The original LaTeX is retained below for exact traceability.

The appendix lists the hardware and software behind every result, maps the artifact repository, and gives one command per result family. The complete artifact is released alongside the paper as an anonymized repository (its README carries the full command map and the exact seeds); model weights are not redistributed, and the artifact downloads them from their official sources. All commands run from the repository root with `PYTHONPATH=src`.

### Hardware and software

One workstation: Windows 11 Pro, Intel Core i9-14900K (32 threads), 192 GB DDR5, an NVIDIA RTX 6000 Ada Generation (48 GB) and an NVIDIA GeForce RTX 4090 (24 GB). The FPGA is an AMD ZCU102 evaluation board (XCZU9EG, `xczu9eg-ffvb1156-2-e`) reached through its Digilent USB-JTAG port; only volatile configuration of the programmable logic was used, and the designs derive their 250 or 300 MHz kernel clock from the board's 125 MHz oscillator with an on-chip MMCM. Software: Python 3.14 with numpy 2.5, matplotlib and openpyxl; CuPy 14.2 (cupy-cuda13x) with NVRTC from pip wheels (no CUDA toolkit); gcc 6.2 (MinGW-w64) and MSVC 19.51; Vivado 2022.2; OpenSSL for the Ed25519 cross-check.

### Repository

`spec/` holds the threat model, relation, manifest schema, leakage policy, numerical semantics, SumFold kernel contract and security obligations. `src/edgecert/` holds the golden model (`field.py`, `sumfold.py`), the integer reference and TileContract compiler (`reference/`, `compiler/`), the runtime, session layer and CertSched (`runtime/`), commitments and signatures (`crypto/`), the independent verifier (`verifier/`), the audit log (`audit/`) and the prover adapters (`prover_adapters/`). `kernels/` holds the C and CUDA kernels; `fpga/` the RTL, testbenches, netlist and replay simulation, build scripts and board host scripts; `bench/` the runners, the attack suite and the generators of every chart, table and number (`bench/analysis/`). Raw result records live under `results/raw/`, one directory per result family, and derived tables under `results/processed/`. The file `paper/generated/numbers.tex` defines every number the text uses as a macro, with its source file, field and evidence class; the artifact's `bench/analysis/loc_count.py` reproduces the component line counts (29,349 lines in 141 files, of which the golden model is 290 lines).

### Commands

The artifact README maps every result family to its exact command, seeds and output files. In brief: unit tests run with `python -m unittest discover -s tests -p "test_*.py"` (185 tests, OK); contracts, real models, scheduling, kernels, the GPT-2 linear-layer prover and the attack suite live under `bench/` (`contracts_eval.py`, `realmodel_*.py`, `sched_*.py`, `kernel_*.py`, `gpt2_linear_prove.py`, `attacks/run_attacks.py`); RTL and netlist simulation under `fpga/sim/`; implementation under `fpga/scripts/launch_impl.py` plus `collect_impl.py`; the board campaign under `fpga/host/board_campaign.py` (needs the ZCU102); and `bench/analysis/make_all.py --latex-check` regenerates every chart, table and number.

### Ethics considerations

The work involves no human subjects and no personal data. We used publicly released models under their licences (GPT-2 under MIT, Qwen3-0.6B under Apache-2.0), synthetic compact models and self-authored prompts; all attacks target our own prototype on our own hardware, and the ZCU102 was configured only volatilely over JTAG. There is no vulnerability disclosure concern.

## Attack and enforcement matrix

This is the complete attack and enforcement matrix formerly printed as Table 6 in the paper. Its source is `paper/tables/gen_attacks.tex`, generated by `bench/analysis/make_tables.py` from `results/raw/attacks/matrix.json`; the source timestamp is `2026-09-29T22:01:21-05:00`. The table is preserved here when removed from the paper.

Each negative case is an attack on the model, numerics, session, hardware or audit path; the positive controls are honest runs. **Expected:** the outcome the policy requires (reject, abort, or uncertified output). **Pass:** observed = expected. **Released:** outputs handed to the actuator. **Total:** 41 cases, 37 negatives, 41 pass, 0 negatives released. P04 (audit log) has no output to release, so it is accepted but not counted as released.

| Category | Cases | Expected | Pass | Released | Detecting checks |
| :--- | :--- | :--- | ---: | ---: | :--- |
| positive control | P01--P04 | 4 accept | 4/4 ✓ | 3 | `--` |
| model binding | A01, A02, A03, A06, A07, A08, A09 | 7 reject | 7/7 ✓ | 0 | `model_commitment_authorized, bad_signature (+3)` |
| enforcement | A31--A35 | 2 abort, 2 reject, 1 uncertified | 5/5 ✓ | 0 | `proof_presence, proof_deadline (+3)` |
| hardware fault | A26--A30 | 4 reject, 1 abort | 5/5 ✓ | 0 | `round_consistency, host_round_self_check` |
| numerical semantics | A10--A14 | 5 reject | 5/5 ✓ | 0 | `remainder_range, membership (+1)` |
| session binding | A04, A05, A18, A19, A20 | 5 reject | 5/5 ✓ | 0 | `rollback_stale_epoch, epoch (+3)` |
| input/context binding | A15--A17 | 3 reject | 3/3 ✓ | 0 | `embedding_position_mask, commitment_opening` |
| sequence coverage | A21--A23 | 3 reject | 3/3 ✓ | 0 | `termination_eos_length, argmax_tiebreak` |
| audit | A36--A37 | 2 reject | 2/2 ✓ | 0 | `rollback, fork` |
| state integrity | A24--A25 | 2 reject | 2/2 ✓ | 0 | `anchor_opening, untrusted_verifier_key` |

<details>
<summary>Original attack-matrix LaTeX (complete source before removal)</summary>

```latex
% AUTO-GENERATED by bench/analysis/make_tables.py from results/ -- do not edit by hand.
% generated 2026-09-29T22:01:21-05:00; sources: results/raw/attacks/matrix.json
\begin{table*}[t]
\caption{\textbf{Attack and enforcement matrix.} Each negative case is an attack on
the model, numerics, session, hardware or audit path; the positive controls are honest runs. Expected: the
outcome the policy requires (reject, abort, or uncertified output); pass: observed $=$ expected; released:
outputs handed to the actuator. Total: 41 cases, 37 negatives, 41 pass, 0 negatives released. P04 (audit log) has no output to release, so it is accepted but not counted as released.}
\label{tab:attacks}
\centering\footnotesize
\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.05}
\begin{tabularx}{\textwidth}{@{}l l l r r >{\raggedright\arraybackslash}X@{}}
\toprule
\textbf{Category} & \textbf{Cases} & \textbf{Expected} & \textbf{Pass} & \textbf{Rel.} & \textbf{Detecting checks}\\
\midrule
positive control & P01--P04 & 4 accept & 4/4\,\cmark & 3 & {\scriptsize\texttt{--}}\\
model binding & A01, A02, A03, A06, A07, A08, A09 & 7 reject & 7/7\,\cmark & 0 & {\scriptsize\texttt{model\_commitment\_authorized, bad\_signature (+3)}}\\
enforcement & A31--A35 & 2 abort, 2 reject, 1 uncertified & 5/5\,\cmark & 0 & {\scriptsize\texttt{proof\_presence, proof\_deadline (+3)}}\\
hardware fault & A26--A30 & 4 reject, 1 abort & 5/5\,\cmark & 0 & {\scriptsize\texttt{round\_consistency, host\_round\_self\_check}}\\
numerical semantics & A10--A14 & 5 reject & 5/5\,\cmark & 0 & {\scriptsize\texttt{remainder\_range, membership (+1)}}\\
session binding & A04, A05, A18, A19, A20 & 5 reject & 5/5\,\cmark & 0 & {\scriptsize\texttt{rollback\_stale\_epoch, epoch (+3)}}\\
input/context binding & A15--A17 & 3 reject & 3/3\,\cmark & 0 & {\scriptsize\texttt{embedding\_position\_mask, commitment\_opening}}\\
sequence coverage & A21--A23 & 3 reject & 3/3\,\cmark & 0 & {\scriptsize\texttt{termination\_eos\_length, argmax\_tiebreak}}\\
audit & A36--A37 & 2 reject & 2/2\,\cmark & 0 & {\scriptsize\texttt{rollback, fork}}\\
state integrity & A24--A25 & 2 reject & 2/2\,\cmark & 0 & {\scriptsize\texttt{anchor\_opening, untrusted\_verifier\_key}}\\
\bottomrule
\end{tabularx}
\end{table*}
```

</details>
