#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PATH="${HOME}/riscv/bin:${PATH}"
MODE="${1:-b0}"
case "$MODE" in
    0|b0) MODE_NAME=b0; MODE=0 ;;
    1|b1) MODE_NAME=b1; MODE=1 ;;
    2|x1) MODE_NAME=x1; MODE=2 ;;
    3|x2) MODE_NAME=x2; MODE=3 ;;
    *) echo "usage: $0 [b0|b1|x1|x2]" >&2; exit 2 ;;
esac

mkdir -p build
python3 scripts/make_input_header.py \
    model/test_vectors/resnet18/input_0.bin build/resnet18_input.h

riscv64-unknown-elf-gcc \
    -std=c11 -march=rv64imc -mabi=lp64 -mcmodel=medany -O2 \
    -ffreestanding -fno-builtin -Wall -Wextra -Werror \
    -nostdlib -nostartfiles -T src/benchmarks/linker.ld \
    -DRESNET18_EXECUTION_MODE="$MODE" \
    -I build \
    src/benchmarks/startup.S \
    src/benchmarks/resnet18_b0_main.c \
    src/benchmarks/resnet18_b0_executor.c \
    model/exported/resnet18/resnet18_graph.c \
    src/kernels/qgemm.c \
    -o "build/resnet18_${MODE_NAME}.elf"

riscv64-unknown-elf-size -A "build/resnet18_${MODE_NAME}.elf"
