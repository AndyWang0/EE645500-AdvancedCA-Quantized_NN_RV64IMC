
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPIKE_BIN="${SPIKE_BIN:-/home/andy/src/riscv-isa-sim/build/spike}"
SPIKE_TIMEOUT="${SPIKE_TIMEOUT:-3600}"

# Modes can be specified as:
#   ./scripts/benchmark.sh x1
#   ./scripts/benchmark.sh "b0 x1"
#   MODES="x1 x2" ./scripts/benchmark.sh
# Default: run all configurations.
MODES_SPEC="${1:-${MODES:-b0 b1 x1 x2}}"

# Support both comma-separated and space-separated modes.
MODES_SPEC="${MODES_SPEC//,/ }"
read -r -a MODES_ARRAY <<< "$MODES_SPEC"

SKIP_SPIKE="${SKIP_SPIKE:-0}"

# Validate configuration.
if ((${#MODES_ARRAY[@]} == 0)); then
    echo "Error: No benchmark modes specified." >&2
    exit 1
fi

case "$SKIP_SPIKE" in
    0|1) ;;
    *)
        echo "Error: SKIP_SPIKE must be 0 or 1." >&2
        exit 1
        ;;
esac

declare -A SEEN_MODES=()

for mode in "${MODES_ARRAY[@]}"; do
    case "$mode" in
        b0|b1|x1|x2) ;;
        *)
            echo "Error: Invalid mode '$mode'. Valid modes: b0 b1 x1 x2." >&2
            exit 1
            ;;
    esac

    if [[ -n "${SEEN_MODES[$mode]:-}" ]]; then
        echo "Error: Duplicate mode '$mode'." >&2
        exit 1
    fi

    SEEN_MODES[$mode]=1
done

cd "$ROOT_DIR"
mkdir -p build results/raw results/tables

echo "Benchmark modes: ${MODES_ARRAY[*]}"
echo "SKIP_SPIKE: $SKIP_SPIKE"

# Build and benchmark selected configurations.
for mode in "${MODES_ARRAY[@]}"; do
    echo
    echo "===== Building ResNet-18 ${mode^^} ====="

    ./scripts/build_resnet18_b0.sh "$mode" \
        >"results/raw/resnet18_${mode}.build.txt"

    riscv64-unknown-elf-size -A "build/resnet18_${mode}.elf" \
        >"results/raw/resnet18_${mode}.size.txt"

    riscv64-unknown-elf-objdump -d "build/resnet18_${mode}.elf" \
        >"results/raw/resnet18_${mode}.disassembly.txt"

    if [[ "$SKIP_SPIKE" -eq 0 ]]; then
        echo "===== Running ResNet-18 ${mode^^} on Spike ====="

        set +e
        timeout "$SPIKE_TIMEOUT" "$SPIKE_BIN" --isa=RV64IMC -g \
            "build/resnet18_${mode}.elf" \
            >"results/raw/resnet18_${mode}.stdout" \
            2>"results/raw/resnet18_${mode}.histogram"
        status=$?
        set -e

        if [[ "$status" -ne 0 ]]; then
            echo "ResNet ${mode} Spike run failed with status ${status}" >&2
            exit "$status"
        fi

        echo "===== Analyzing ResNet-18 ${mode^^} ====="

        python3 scripts/analyze_histogram.py \
            "results/raw/resnet18_${mode}.histogram" \
            "results/raw/resnet18_${mode}.disassembly.txt" \
            "results/tables/resnet18_${mode}.csv"
    else
        echo "Skipping Spike execution and histogram analysis for ${mode^^}."
    fi
done

# Generate a summary only when fresh Spike results are available.
if [[ "$SKIP_SPIKE" -eq 0 ]]; then
    python3 - "${MODES_ARRAY[@]}" <<'PY'
import csv
import sys
from pathlib import Path

root = Path("results")
names = sys.argv[1:]

# Load results for selected configurations only.
rows = {}

for name in names:
    csv_path = root / "tables" / f"resnet18_{name}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing benchmark result: {csv_path}")

    with csv_path.open(newline="") as stream:
        rows[name] = next(csv.DictReader(stream))

# Use B0 as the baseline only when B0 was benchmarked.
baseline = (
    int(rows["b0"]["dynamic_instructions"])
    if "b0" in rows else None
)

# Construct summary columns.
derived_fields = [
    "instructions_per_output",
    "macs_per_instruction",
    "text_bytes",
    "instruction_change_vs_b0",
]

source_fields = [
    field for field in rows[names[0]]
    if field not in ("region", "configuration", *derived_fields)
]

fields = ["configuration"] + source_fields + derived_fields

summary_path = root / "tables" / "resnet18_summary.csv"

with summary_path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()

    for name in names:
        row = rows[name]

        instructions = int(row["dynamic_instructions"])
        macs = int(row["logical_macs"])

        output = {
            key: value
            for key, value in row.items()
            if key != "region"
        }

        output["configuration"] = name.upper()

        output["instructions_per_output"] = (
            f"{instructions / 1000:.4f}"
        )

        output["macs_per_instruction"] = (
            f"{macs / instructions:.6f}"
            if instructions else "N/A"
        )

        # Extract .text size from the corresponding ELF size report.
        size_path = root / "raw" / f"resnet18_{name}.size.txt"

        with size_path.open() as size_stream:
            text_bytes = next(
                (
                    line.split()[1]
                    for line in size_stream
                    if line.split() and line.split()[0] == ".text"
                ),
                None,
            )

        if text_bytes is None:
            raise ValueError(f"Could not find .text size in {size_path}")

        output["text_bytes"] = text_bytes

        if baseline is not None and baseline != 0:
            output["instruction_change_vs_b0"] = (
                f"{(baseline - instructions) / baseline:.4%}"
            )
        else:
            output["instruction_change_vs_b0"] = "N/A"

        writer.writerow(output)

print(f"Summary written to {summary_path}")
PY

else
    echo
    echo "Spike was skipped. Summary CSV was not regenerated."
fi

echo
echo "===== Benchmark completed ====="
echo "Selected configurations: ${MODES_ARRAY[*]}"

if [[ "$SKIP_SPIKE" -eq 0 ]]; then
    echo "Summary: results/tables/resnet18_summary.csv"
else
    echo "Build, size, and disassembly outputs are available in results/raw/."
fi
