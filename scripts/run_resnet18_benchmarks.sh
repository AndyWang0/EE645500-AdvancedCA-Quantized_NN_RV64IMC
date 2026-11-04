#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPIKE_BIN="${SPIKE_BIN:-/home/andy/src/riscv-isa-sim/build/spike}"
SPIKE_TIMEOUT="${SPIKE_TIMEOUT:-3600}"
LOGICAL_MACS=1814073344
cd "$ROOT_DIR"
mkdir -p build results/raw results/tables

for mode in b0 b1 x1 x2; do
    ./scripts/build_resnet18_b0.sh "$mode" >"results/raw/resnet18_${mode}.build.txt"
    riscv64-unknown-elf-objdump -d "build/resnet18_${mode}.elf" \
        >"results/raw/resnet18_${mode}.disassembly.txt"
    set +e
    timeout "$SPIKE_TIMEOUT" "$SPIKE_BIN" --isa=RV64IMC \
        --log="results/raw/resnet18_${mode}.trace" -l \
        "build/resnet18_${mode}.elf" \
        >"results/raw/resnet18_${mode}.stdout" \
        2>"results/raw/resnet18_${mode}.stderr"
    status=$?
    set -e
    if [ "$status" -ne 0 ]; then
        echo "ResNet ${mode} Spike run failed with status ${status}" >&2
        exit "$status"
    fi
    python3 scripts/analyze_trace.py \
        "results/raw/resnet18_${mode}.trace" \
        "results/raw/resnet18_${mode}.disassembly.txt" \
        "results/tables/resnet18_${mode}.csv" \
        "$LOGICAL_MACS"
done

python3 - <<'PY'
import csv
from pathlib import Path

root = Path("results")
names = ("b0", "b1", "x1", "x2")
rows = {}
for name in names:
    with (root / "tables" / f"resnet18_{name}.csv").open(newline="") as stream:
        rows[name] = next(csv.DictReader(stream))
baseline = int(rows["b0"]["dynamic_instructions"])
fields = list(rows["b0"])
with (root / "tables" / "resnet18_summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for name in names:
        row = rows[name]
        instructions = int(row["dynamic_instructions"])
        macs = int(row["logical_macs"])
        row["configuration"] = name.upper()
        row["instructions_per_output"] = ""
        row["macs_per_instruction"] = f"{macs / instructions:.6f}"
        row["text_bytes"] = ""
        row["instruction_change_vs_b0"] = f"{(baseline - instructions) / baseline:.4%}"
        writer.writerow(row)
PY

echo "ResNet-18 B0/B1/X1/X2 benchmark results written to results/tables/resnet18_summary.csv"
