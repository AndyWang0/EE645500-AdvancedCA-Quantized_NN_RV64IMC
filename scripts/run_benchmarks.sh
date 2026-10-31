#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPIKE_DIR="${SPIKE_DIR:-${HOME}/src/riscv-isa-sim}"
cd "$ROOT_DIR"
mkdir -p build results/raw results/tables

COMMON_FLAGS=(
    -std=c11 -march=rv64imc -mabi=lp64 -mcmodel=medany -O2
    -ffreestanding -fno-builtin -Wall -Wextra -Werror
    -nostdlib -nostartfiles -T src/benchmarks/linker.ld
)

for mode in 0 1 2; do
    case "$mode" in
        0) name=b0 ;;
        1) name=x1 ;;
        2) name=x2 ;;
    esac
    riscv64-unknown-elf-gcc "${COMMON_FLAGS[@]}" -DBENCHMARK_MODE="$mode" \
        src/benchmarks/startup.S src/benchmarks/benchmark_main.c \
        src/kernels/qgemm.c -o "build/${name}.elf"
    riscv64-unknown-elf-size -A "build/${name}.elf" \
        > "results/raw/${name}.size.txt"
    riscv64-unknown-elf-objdump -d "build/${name}.elf" \
        > "results/raw/${name}.disassembly.txt"
    "$SPIKE_DIR/build/spike" --isa=rv64imc --instructions=200000 \
        --log="results/raw/${name}.trace" -l "build/${name}.elf" \
        > "results/raw/${name}.stdout" 2> "results/raw/${name}.stderr" || true
    if grep -q 'c.ebreak' "results/raw/${name}.trace"; then
        echo "${name} output verification failed" >&2
        exit 1
    fi
    python3 scripts/analyze_trace.py \
        "results/raw/${name}.trace" \
        "results/raw/${name}.disassembly.txt" \
        "results/tables/${name}.csv"
done

python3 - <<'PY'
import csv
from pathlib import Path

rows = {}
for name in ("b0", "x1", "x2"):
    with open(Path("results/tables") / f"{name}.csv", newline="") as stream:
        rows[name] = next(csv.DictReader(stream))

fields = [
    "configuration", "dynamic_instructions", "loads", "stores", "integer",
    "multiply", "branches", "custom", "xqmac8", "xqrequant8",
    "logical_macs", "instructions_per_output", "macs_per_instruction",
    "text_bytes", "instruction_change_vs_b0",
]
with open("results/tables/summary.csv", "w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for name in ("b0", "x1", "x2"):
        row = rows[name]
        instructions = int(row["dynamic_instructions"])
        logical_macs = int(row["logical_macs"])
        with open(Path("results/raw") / f"{name}.size.txt") as size_stream:
            text_bytes = next(
                int(line.split()[1])
                for line in size_stream
                if line.split() and line.split()[0] == ".text"
            )
        row["configuration"] = name.upper()
        row["instructions_per_output"] = f"{instructions / 128:.4f}"
        row["macs_per_instruction"] = f"{logical_macs / instructions:.6f}"
        row["text_bytes"] = text_bytes
        row["instruction_change_vs_b0"] = "0.0000"
        if name != "b0":
            baseline = int(rows["b0"]["dynamic_instructions"])
            row["instruction_change_vs_b0"] = (
                f"{(baseline - instructions) / baseline:.4%}"
            )
        writer.writerow({field: row[field] for field in fields})
PY

echo "Benchmark results written to results/tables/summary.csv"
