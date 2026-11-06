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
    -O3 \
    -ffreestanding \
    -fno-builtin \
    -Wall -Wextra -Werror \
    -nostdlib -nostartfiles \
    -T src/benchmarks/linker.ld \
    src/benchmarks/startup.S \
    tests/test_custom_spike.c \
    -o build/custom_spike_test.elf

"$SPIKE_DIR/build/spike" \
    --isa=rv64imc \
    --instructions=100000 \
    --log=results/raw/custom_spike_test.trace \
    -l \
    build/custom_spike_test.elf \
    > results/raw/custom_spike_test.stdout \
    2> results/raw/custom_spike_test.stderr || true

XQMAC_COUNT="$(grep -c 'xqmac8' results/raw/custom_spike_test.trace || true)"
XQREQUANT_COUNT="$(grep -c 'xqrequant8' results/raw/custom_spike_test.trace || true)"

if [ "$XQMAC_COUNT" -ge 67 ] &&
   [ "$XQREQUANT_COUNT" -ge 67 ] &&
   grep -q 'c.nop' results/raw/custom_spike_test.trace &&
   ! grep -q 'c.ebreak' results/raw/custom_spike_test.trace; then
    echo "custom Spike instruction test passed (${XQMAC_COUNT} xqmac8, ${XQREQUANT_COUNT} xqrequant8)"
    exit 0
fi

echo "custom Spike instruction test failed" >&2
exit 1
