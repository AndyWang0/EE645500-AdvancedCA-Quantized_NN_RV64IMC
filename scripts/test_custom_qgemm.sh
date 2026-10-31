#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPIKE_DIR="${SPIKE_DIR:-${HOME}/src/riscv-isa-sim}"
cd "$ROOT_DIR"

mkdir -p build results/raw

riscv64-unknown-elf-gcc \
    -std=c11 \
    -march=rv64imc \
    -mabi=lp64 \
    -mcmodel=medany \
    -O2 \
    -ffreestanding \
    -fno-builtin \
    -Wall -Wextra -Werror \
    -nostdlib -nostartfiles \
    -T src/benchmarks/linker.ld \
    src/benchmarks/startup.S \
    src/benchmarks/x1_main.c \
    src/kernels/qgemm.c \
    -o build/x1.elf

riscv64-unknown-elf-objdump -d build/x1.elf \
    > results/raw/x1.disassembly.txt

"$SPIKE_DIR/build/spike" \
    --isa=rv64imc \
    --instructions=200000 \
    --log=results/raw/x1.trace \
    -l \
    build/x1.elf \
    > results/raw/x1.stdout \
    2> results/raw/x1.stderr || true

if grep -q 'xqmac8' results/raw/x1.trace &&
   grep -q 'xqrequant8' results/raw/x1.trace &&
   grep -q 'c.nop' results/raw/x1.trace &&
   ! grep -q 'c.ebreak' results/raw/x1.trace; then
    echo "custom qgemm Spike test passed"
    exit 0
fi

echo "custom qgemm Spike test failed" >&2
exit 1
