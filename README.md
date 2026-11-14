# EDGE-CERT: lightweight, memory-bounded, hardware-accelerated certification of compact language models

Submission artifact: anonymized. It contains the code that implements the four core modules of the paper and the experiment harness behind the reported results.

- `TILECONTRACT`: tile-level range/accumulator contracts, static certification and accounting.
- `CERTSCHED`: commit/challenge-aware witness scheduling and the discrete-event simulator.
- `SUMFOLD`: fused product-sumcheck kernel (C/OpenMP, CUDA) and the Zynq RTL/bitstream flow.
- Signed manifest contracts: Ed25519-signed model manifests, sessions, honest verifier, certificates.

The artifact is code-only: no model weights, no bitstreams and no paper copy are redistributed. Weights are fetched from their public upstreams (`scripts/get_models.py`, SHA-256 pinned); FPGA flows are described in `AE.md` and run with Vivado plus a ZCU102 board.

## Layout

| Path | Contents |
| :--- | :--- |
| `src/edgecert/compiler/` | TILECONTRACT: graph lowering, gadgets, contracts, census, accounting |
| `src/edgecert/runtime/` | CERTSCHED: DAG, scheduler, simulator, signed manifests, sessions |
| `src/edgecert/sumfold.py`, `kernels/` | SUMFOLD: golden model, CPU kernel, CUDA kernel, host shim |
| `src/edgecert/verifier/`, `src/edgecert/audit/` | honest verifier, transcripts, receipts |
| `src/edgecert/prover_adapters/` | transparent reference prover, MLE/sumcheck adapters, toy model |
| `fpga/` | RTL (`rtl/`), testbenches (`tb/`), constraints, Vivado scripts, board hosts, golden vectors |
| `bench/` | experiment runners and analysis (contracts, kernels, scheduling, models, attacks, tables) |
| `spec/`, `configs/`, `data/` | interface and kernel specs, cost profile, prompt sets |
| `demo/` | self-contained end-to-end demo (Python) and CPU-kernel demo (make) |
| `tests/` | the verification-side unit tests needed to check a fresh checkout |

## Requirements

Python 3.10+ with `numpy` and `pyyaml` (see `requirements.txt`). The CPU kernel needs `gcc` (or `cc`) with OpenMP and `cmake` (or the shipped `Makefile`). CUDA kernels need a CUDA toolkit and the GPU runs use CuPy; both are optional (CPU fallbacks exist). The FPGA flow needs Vivado 2022.2 and a ZCU102 board; nothing about it is bundled as binaries.

## Quickstart

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
python demo/demo.py              # manifest -> TileContract -> CertSched -> prove -> verify
./demo/kernel_demo.sh            # build the CPU SumFold kernel and run its self-test
python -m unittest discover -s tests -p "test_*.py"   # verification-side unit tests
```

Docker: `docker build -t edgecert . && docker run --rm edgecert`.

## Notes

- Timing values in the paper were measured on specific hosts.
- All absolute paths in the code are placeholders (`<artifact-root>`, `<home>`); the artifact contains no author names, affiliations or contact addresses.
- Licence: MIT (`LICENSE`). Upstream model licences apply to the downloaded weights.

See `AE.md` for the evaluation guide and `ARCHITECTURE.md` / `INTERFACES.md` for the design and the interfaces between the components.
