#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PATH="${HOME}/riscv/bin:${PATH}"

mkdir -p build results/raw

riscv64-unknown-elf-gcc \
    -std=c11 \
    -march=rv64imc \
    -mabi=lp64 \
    -mcmodel=medany \
    -O3 \
    -ffreestanding \
    -fno-builtin \
    -Wall -Wextra -Werror \
    -nostdlib -nostartfiles \
    -T src/benchmarks/linker.ld \
    src/benchmarks/startup.S \
    src/benchmarks/b0_main.c \
    src/kernels/qgemm.c \
    -o build/b0.elf

riscv64-unknown-elf-size -A build/b0.elf > results/raw/b0.size.txt
riscv64-unknown-elf-objdump -d build/b0.elf > results/raw/b0.disassembly.txt

SPIKE_BIN="$(command -v spike || echo "${HOME}/riscv/bin/spike")"
"${SPIKE_BIN}" \
    --isa=rv64imc \
    --instructions=200000 \
    --log=results/raw/b0.trace \
    -l \
    build/b0.elf \
    > results/raw/b0.stdout 2> results/raw/b0.stderr || true

DYNAMIC_INSTRS="$(grep -c '0x000000' results/raw/b0.trace || true)"
TEXT_SIZE="$(awk '$1==".text" {print $2}' results/raw/b0.size.txt)"
RODATA_SIZE="$(awk '$1==".rodata" {print $2}' results/raw/b0.size.txt)"
DATA_SIZE="$(awk '$1==".data" {print $2}' results/raw/b0.size.txt)"
BSS_SIZE="$(awk '$1==".bss" {print $2}' results/raw/b0.size.txt)"

cat > results/raw/b0.summary.txt <<EOF
B0 benchmark summary
ELF: build/b0.elf
Dynamic instructions: ${DYNAMIC_INSTRS}
.text size: ${TEXT_SIZE}
.rodata size: ${RODATA_SIZE}
.data size: ${DATA_SIZE}
.bss size: ${BSS_SIZE}
Trace: results/raw/b0.trace
Disassembly: results/raw/b0.disassembly.txt
EOF

echo "B0 benchmark built and measured. Summary written to results/raw/b0.summary.txt"
