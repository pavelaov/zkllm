# SumFold kernels (CPU C, CUDA)

Implementations of the SumFold product sumcheck (`spec/sumfold_kernel.md` v1). All of them are bit-exact with the golden model `src/edgecert/sumfold.py::prove()` (claim, every round message, every challenge, final `f~(r)`, `g~(r)`, as 128-bit hex words `{c1,c0}`).

| Path | What |
|---|---|
| `cpu/gl_field.h` | Goldilocks mul (`unsigned __int128` or MSVC `_umul128`), multiplier-free `reduce128` exactly as `field.reduce128`, branch-free corrections, Ext2 Karatsuba, lazy 128-bit accumulators |
| `cpu/sha256.[ch]` | FIPS 180-4 SHA-256 for the transcript |
| `cpu/sumfold.[ch]` | fused engine (spec 3.1 + lanes of 3.2, in place, host tail) and unfused engine (evaluate pass + out-of-place ping-pong fold pass) |
| `cpu/main.c` | `sumfold_cpu` CLI, JSON output |
| `cuda/sumfold.cu` | CUDA kernels `sf_eval`, `sf_fold_eval` (fused), `sf_fold`, `sf_reduce_partials`; NVRTC-ready, ping-pong buffers (see the header comment for why not in place) |
| `cuda/shim/` | SIMT host shim: compiles the unmodified `sumfold.cu` with g++ and executes it on the CPU (OpenMP team per block) for bit-exactness tests without a CUDA toolkit |

## Build

The runners build automatically (`bench/runners/kernel_common.build_cpu`). Manual commands:

```
set PATH=<vivado-2022.2>\tps\mingw\6.2.0\win64.o\nt\bin;%PATH%
gcc -O3 -march=native -funroll-loops -fopenmp -static -std=gnu11 -Wall ^
    -o kernels\cpu\build\sumfold_cpu.exe kernels\cpu\main.c kernels\cpu\sumfold.c kernels\cpu\sha256.c
```

MSVC (VS BuildTools 18, from a `vcvars64.bat` prompt):

```
cl /O2 /openmp /nologo /W3 /D_CRT_SECURE_NO_WARNINGS /Fe:kernels\cpu\build\sumfold_cpu_msvc.exe ^
   kernels\cpu\main.c kernels\cpu\sumfold.c kernels\cpu\sha256.c
```

Linux: `make -C kernels/cpu` (gcc, OpenMP).

CUDA shim DLL (correctness only):

```
g++ -O2 -std=c++14 -fopenmp -shared -static-libgcc -static-libstdc++ -Wno-unknown-pragmas ^
    -o kernels\cuda\build\sumfold_shim.dll kernels\cuda\shim\shim_launch.cpp
```

On a machine with CuPy, `bench/runners/kernel_gpu_bench.py` compiles `sumfold.cu` with NVRTC. With the CUDA toolkit, `nvcc -arch=sm_89 -cubin kernels/cuda/sumfold.cu` also works (device code only).

## CLI

```
sumfold_cpu --logn n --threads t --variant fused|unfused --seedf HEX --seedg HEX
            --reps R --warmup W --out file.json [--lanes L] [--sched static|dynamic] [--affinity HEXMASK]
sumfold_cpu --selftest
```

Inputs are the spec 3.3 xorshift64 test vectors. Tables are re-expanded (untimed) before each of the `W + R` runs; `total_s` covers every pass, the SHA-256 transcript and the host tail.

## Tests and benchmarks

```
python -m unittest tests.test_kernels_cpu tests.test_kernels_gpu_emu -v
python bench/runners/kernel_cpu_bench.py [--reduced|--auto-reduce] [--compilers gcc,msvc] [--pin pfirst|none]
python bench/runners/kernel_gpu_bench.py          # NOT_RUN without CuPy, plus emulation checks
python bench/runners/kernel_opcount.py            # ANALYTIC op/traffic counts
```

Results: `results/raw/kernels/{cpu_bench,gpu_bench,opcount}.json`. Notes: `docs/notes/kernels.md`.
