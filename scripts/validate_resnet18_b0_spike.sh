#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
SPIKE_BIN="${SPIKE_BIN:-/home/andy/src/riscv-isa-sim/build/spike}"
SPIKE_INSTRUCTIONS="${SPIKE_INSTRUCTIONS:-0}"
SPIKE_TIMEOUT="${SPIKE_TIMEOUT:-1800}"
mkdir -p build results/raw

python3 scripts/make_input_header.py \
    model/test_vectors/resnet18/input_0.bin build/resnet18_input.h
python3 scripts/make_tensor_reference_header.py \
    model/test_vectors/resnet18/validation_metadata.json \
    build/resnet18_reference_tensors.h

riscv64-unknown-elf-gcc \
    -std=c11 -march=rv64imc -mabi=lp64 -mcmodel=medany -O2 \
    -ffreestanding -fno-builtin -Wall -Wextra -Werror \
    -nostdlib -nostartfiles -T src/benchmarks/linker.ld \
    -I build -I model/exported/resnet18 \
    src/benchmarks/startup.S \
    src/benchmarks/resnet18_validation_main.c \
    src/benchmarks/resnet18_b0_executor.c \
    src/kernels/qgemm.c \
    -o build/resnet18_b0_validation.elf

SPIKE_ARGS=(--isa=RV64IMC)
if [ "$SPIKE_INSTRUCTIONS" -gt 0 ]; then
    SPIKE_ARGS+=(--instructions="$SPIKE_INSTRUCTIONS")
fi

set +e
timeout "$SPIKE_TIMEOUT" "$SPIKE_BIN" "${SPIKE_ARGS[@]}" \
    build/resnet18_b0_validation.elf \
    >results/raw/resnet18_b0_validation.stdout \
    2>results/raw/resnet18_b0_validation.stderr
SPIKE_STATUS=$?
set -e

if [ "$SPIKE_STATUS" -eq 0 ]; then
    echo "Spike B0 validation passed"
    exit 0
fi

if [ "$SPIKE_STATUS" -eq 124 ]; then
    echo "Spike validation timed out; the model did not report tohost" >&2
    exit 124
fi

if grep -q '^RESNET18_FAIL ' results/raw/resnet18_b0_validation.stdout; then
    cat results/raw/resnet18_b0_validation.stdout
    exit 1
fi

echo "Spike validation terminated unexpectedly with status $SPIKE_STATUS" >&2
cat results/raw/resnet18_b0_validation.stderr >&2
exit 2
