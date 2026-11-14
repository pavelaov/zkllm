#!/usr/bin/env bash
# Build the CPU SumFold kernel and run its self-test (no Python needed).
set -euo pipefail
cd "$(dirname "$0")/.."
make -C kernels/cpu -s build/sumfold_cpu
./kernels/cpu/build/sumfold_cpu --selftest
