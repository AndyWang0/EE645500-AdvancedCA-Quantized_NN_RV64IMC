#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PATH="${HOME}/riscv/bin:${PATH}"

mkdir -p build
python3 scripts/make_input_header.py \
    model/test_vectors/resnet18/input_0.bin build/resnet18_input.h

riscv64-unknown-elf-gcc \
    -std=c11 -march=rv64imc -mabi=lp64 -mcmodel=medany -O2 \
    -ffreestanding -fno-builtin -Wall -Wextra -Werror \
    -nostdlib -nostartfiles -T src/benchmarks/linker.ld \
    -I build \
    src/benchmarks/startup.S \
    src/benchmarks/resnet18_b0_main.c \
    src/benchmarks/resnet18_b0_executor.c \
    src/kernels/qgemm.c \
    -o build/resnet18_b0.elf

riscv64-unknown-elf-size -A build/resnet18_b0.elf
