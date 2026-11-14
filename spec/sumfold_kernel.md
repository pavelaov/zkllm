# SumFold kernel contract (v1 semantics, v2 implementation requirements)

Revision 2026-09-29: register CFG clock field corrected to `CFG[31:16]` (section 5, note S1); section 7 added (v2 explicit multiplier and mandatory netlist-level verification). The protocol semantics of sections 2 to 4, the transcript and the golden model are unchanged.

Single source of truth for every SumFold implementation: Python golden model (`src/edgecert/sumfold.py`, `src/edgecert/field.py`), CPU C kernel (`kernels/cpu/`), CUDA kernel (`kernels/cuda/`) and the ZCU102 RTL (`fpga/`). All implementations MUST be bit-exact with the golden model. 

## 1. Role in EDGE-CERT

The product sumcheck `H = sum_b f~(b) g~(b)` is the inner loop of the matrix-multiplication sumcheck (C~(r1,r2) = sum_k A~(r1,k) B~(k,r2)) and of LogUp-style lookup/range accumulators used by sumcheck/GKR backends. Its working set (two evaluation tables) is re-read in every round, so it is the natural *resident* kernel for a board with on-chip memory: load once, fold `n` times, return `O(n)` field elements. This is the FPGA candidate named in the design report (Section 7.2): "finite-field vector folding / sumcheck reduction".

## 2. Arithmetic

* Base field: Goldilocks `p = 2^64 - 2^32 + 1`.
* Challenge/extension field: `Ext = F_p[X]/(X^2 - 7)`; element `(c0, c1)`. Storage word = 128 bits `{c1[63:0], c0[63:0]}` (c0 in the low half).
* Base multiply: 64x64 -> 128, then Goldilocks reduction `x = lo - hh + hl*(2^32-1) mod p` with `hi = hh*2^32 + hl` (no multiplier needed for the reduction; see `field.reduce128`).
* Ext multiply (Karatsuba): `t0=a0*b0, t1=a1*b1, t2=(a0+a1)(b0+b1)`, `c0 = t0 + 7*t1`, `c1 = t2 - t0 - t1`. `7*t1` is `(t1<<3) - t1` mod p.
* All values are canonical (`< p`) at every storage/interface boundary.

Security note: |Ext| ~ 2^128, so a single sumcheck round has soundness error <= 2/|Ext| ~ 2^-127. The end-to-end security level of a full proof depends on the backend (commitment, Fiat-Shamir, number of composed claims).

## 3. Protocol semantics (what the kernel computes)

Inputs: `f, g` of length `N = 2^n` (Ext elements; base-field inputs have `c1 = 0`). Variables are bound LSB-first.

Round `i` (on the current tables `f, g` of length `M = N / 2^i`):

```
s0 = sum_j f[2j]   * g[2j]
s1 = sum_j f[2j+1] * g[2j+1]
s2 = sum_j (2 f[2j+1] - f[2j]) * (2 g[2j+1] - g[2j])
r_i = Transcript.round(s0, s1, s2)          # host side, SHA-256 chain
f'[j] = f[2j] + r_i * (f[2j+1] - f[2j])     # same for g
```

Transcript: `state_0 = SHA256("EDGE-CERT/sumfold/v1" || u32le(n) || enc(H))`, `state_{i+1} = SHA256(state_i || enc(s0) || enc(s1) || enc(s2))`, `r_i = (u64le(state_{i+1}[0:8]) mod p, u64le(state_{i+1}[8:16]) mod p)`, `enc(e) = u64le(c0) || u64le(c1)`.

### 3.1 Fused fold-and-evaluate pass (required microarchitecture idea)

A naive implementation makes two passes per round (evaluate, then fold). SumFold fuses them: pass `i >= 1` streams the level-`(i-1)` tables once, applies `fold(r_{i-1})`, and feeds consecutive folded outputs `(f'[2j], f'[2j+1])` directly into the round-`i` accumulators. Pass 0 only evaluates; a final pass applies `fold(r_{R-1})`. Total passes `R + 1`, each element read once per pass. The folded table is written **in place** (`f'[j]` overwrites a slot already consumed, since `j <= 2j`), so no ping-pong buffer is needed: resident memory is exactly `2 * N * 128` bits.

Recommended memory organisation for in-place single-pass operation: each vector split into an EVEN bank (`v[2j]` at address `j`) and an ODD bank (`v[2j+1]` at address `j`); a pass reads address `j` of both banks per cycle and writes `f'[j]` to bank `j & 1`, address `j >> 1` (write address always trails read address).

### 3.2 Lanes

`L` lanes (power of two). Lane `l` owns the contiguous block `[l*N/L, (l+1)*N/L)` of both tables. The device executes rounds `0 .. R-1` with `R = n - log2(L)` (per-lane block >= 2 elements), summing the per-lane partial `(s0, s1, s2)` with an adder tree, then applies the final fold with `r_{R-1}`, leaving one `(f, g)` pair per lane. The host finishes the last `log2(L)` rounds (`sumfold.host_tail`). For `L = 1` the device runs all `n` rounds and returns the final `(f~(r), g~(r))`.

Golden expected values for any `(seed_f, seed_g, n, L)`: `sumfold.golden(seed_f, seed_g, n, L)` -> dict with `claim`, `challenges`, `msgs` (device rounds), `lane_final_f/g`, plus the full-protocol values.

### 3.3 Test vector generator (on-chip self-test mode)

`v[i] = x_{i+1} mod p`, `x_0 = seed` (non-zero), `x_{k+1} = xorshift64(x_k)` with `x ^= x<<13; x ^= x>>7; x ^= x<<17` on 64 bits; `c1 = 0`. The RTL generator fills lane-local memories with exactly this global sequence (lane `l` starts at global index `l*N/L`; implementation may run L generators seeded by pre-computed jump states supplied by the host, or one generator filling sequentially; both must yield identical tables).

## 4. Operating modes

* **AUTO (benchmark replay):** host preloads challenges `r_0..r_{R-1}` (computed by the host transcript over the same data) and the device runs all passes back-to-back. Used to measure pure kernel throughput. Outputs must equal the golden model; this mode never replaces the interactive protocol in a real proof.
* **STEP (interactive):** device runs one pass, host reads `(s0,s1,s2)`, computes `r_i`, writes it, starts the next pass. Measures the per-round host round-trip that a real deployment pays (JTAG here; PS/AXI in the production path).

## 5. Minimum control/status interface (AXI4-Lite, 32-bit words)

| Offset | Name | Meaning |
|---|---|---|
| 0x000 | ID | constant `0xEDCE2026` |
| 0x004 | CFG | `[7:0] L`, `[15:8] log2(N_max)`, `[31:16] build kernel clock MHz` (16-bit field; see note S1) |
| 0x008 | CTRL | bit0 START, bit1 GEN, bit2 SOFT_RESET, bit4 MODE(0=AUTO,1=STEP) |
| 0x00C | STATUS | bit0 BUSY, bit1 DONE, bit2 ERROR (RTL adds bit3 WAIT = STEP message ready, bit4 GEN_DONE) |
| 0x010 | LOGN | n for the current job |
| 0x014 | ROUND | index of the last completed pass |
| 0x018..0x024 | SEED_F lo/hi, SEED_G lo/hi | PRNG seeds |
| 0x040..0x054 | CH_IDX, CH_W0..CH_W3, CH_WE | challenge RAM write port (W0 = c0[31:0] ... W3 = c1[63:32]) |
| 0x060..0x070 | RES_IDX, RES_W0..W3 | read port: IDX = 3*round + k selects s_k; IDX >= 3*16 selects lane finals |
| 0x080..0x08C | CYC_TOTAL lo/hi, CYC_GEN lo/hi | kernel-clock cycle counters |
| 0x090 | CYC_ROUND_IDX, 0x094 CYC_ROUND | per-pass cycle counts |
| 0x0A0..0x0B8 | MEM_ADDR, MEM_SEL, MEM_W0..W3, MEM_WE | direct table access (small exact tests) |

Implementations may add registers.

**Note S1 (corrected 2026-09-29).** Version 1 of this table placed the build clock in `CFG[23:16]`, an 8-bit field that cannot represent 300 MHz. The RTL (`fpga/rtl/sumfold_axil.v`, header comment and `KCLK16`) implements `CFG[31:16] = KCLK_MHZ`, and every board record was read that way (e.g. v1 L8 250 MHz: `CFG = 0x00FA1008`, i.e. L = 8, log2 N_max = 16, 250 MHz; the 300 MHz bitstreams report `CFG[31:16] = 300`). `CFG[31:16]` is now normative. Host software must decode the clock from bits 31:16, never from 23:16.

Register extensions implemented by the RTL: `CH_IDX` range 0..15 (at most 16 device rounds); `RES_IDX = 48 + 2l` (`+1`) selects lane `l`'s final `f` (`g`); `0x098/0x09C CYC_WALL` (START to DONE including STEP host waits, used to measure the kernel clock); `0x0C0..0x0CC MEM_R0..R3` and `0x0D0 MEM_RVALID` (MEM read-back; `MEM_WE` bit1 = read); `0x0F0 KCLK_MHZ` (build kernel clock). `MEM_ADDR` is not cleared by `SOFT_RESET` or `GEN`.

## 6. v2 implementation requirements: explicit multiplier and netlist verification

**Why (lesson of 2026-09-27).** The v1 RTL wrote the 64 x 64 base product as a behavioural `a * b` followed by pipeline registers (now `GL_MUL_LEGACY_INFER` in `fpga/rtl/gl_arith.v`). Vivado 2022.2 decomposed it into a 16-DSP48E2 cascade and redistributed the registers so that two partial products (weights 2^51 and 2^85) arrived one cycle late and one cycle early. The RTL passed all 27 bit-exact `tb_core` cases and `tb_field`, every build closed timing, and control flow and cycle counts were exact on the board, yet every round message was wrong (MEASURED: `results/raw/fpga_board/{bringup_L8_250_auto_n10,v1_L1_250_auto_n10,v1_L8_300_perf_auto_n10}.json`, `VERIFY_FAILED`). Netlist simulation of the synthesized multiplier reproduced the board bit for bit (SIMULATION: `results/raw/fpga_sim/netlist_gl_mul.json`, 505 of 512 streamed products wrong, 8 of 64 impulses misaligned; `replay_*_mulnet*.json`, 52/52, 133/133 and 47/47 values). RTL simulation and static timing cannot see this class of fault. The host's round check rejected every such transcript, as section 3 requires.

**v2 multiplier (normative for the default build).** `gl_mul` is an explicit DSP48E2 decomposition with every pipeline register written in the RTL:

* `a = a0 + a1 2^26 + a2 2^52` (26, 26, 12 bits), `b = b0 + b1 2^17 + b2 2^34 + b3 2^51` (17, 17, 17, 13 bits); each partial product is at most 26 x 17 bits (one DSP48E2).
* Column `i`: 4-stage cascade `P_j = (P_{j-1} >> 17) + a_i b_j` (PCIN>>17), `c_i = {P_3, P_2[16:0], P_1[16:0], P_0[16:0]}`; `prod = c_0 + c_1 2^26 + c_2 2^52`.
* 12 DSP48E2 per base multiplier (v1: 16), 144 per lane (twelve base multipliers per lane: four extension multipliers x 3 Karatsuba products); L = 8 uses 1,152 of 2,520 (POST_ROUTE, `results/raw/fpga_impl/runs.json`).
* Latency unchanged: `GL_MUL_PIPE = 6` (an elaboration error enforces it), `GL_MUL_LAT = 11`, `EXT_MUL_LAT = 18`; every cycle count of `results/raw/fpga_sim/sim_results.json` is unchanged and equals the on-board counters (MEASURED, `results/raw/fpga_board/summary.json`).

**Netlist-verification requirement (mandatory before any bitstream is programmed).** A change to any arithmetic unit (`gl_arith.v`, `ext_arith.v`) or to synthesis options is verified by RTL simulation and timing closure alone. Before a new bitstream is used on the board, all of the following must pass and be recorded under `results/raw/fpga_sim/`:

1. RTL regression: `python fpga/sim/run_sim.py` (27 core cases, field units, AXI, top clocking).
2. Post-synthesis functional netlist of every multiplier variant that is built, with back-to-back (II = 1) streaming vectors and single-bit impulse tests (every partial product at the expected cycle): `python fpga/sim/netlist/run_netlist_checks.py` (`netlist_gl_mul.json`; v2: 512/512 at II = 1, 512/512 with gaps, 64/64 impulses at 11 cycles).
3. Replay of a real board register trace on the RTL and on the out-of-context synthesized `sumfold_axil` netlist: `python fpga/sim/replay/replay.py ...` (`match: true`, `replay_L8_n10_synthnet_v2.json`).
4. On the board, the host verifier (`sumfold.verify` over the Fiat-Shamir transcript) must stay enabled for every job.
