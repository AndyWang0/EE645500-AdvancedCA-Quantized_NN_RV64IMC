#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPIKE_BIN="${SPIKE_BIN:-/home/andy/src/riscv-isa-sim/build/spike}"
SPIKE_TIMEOUT="${SPIKE_TIMEOUT:-3600}"
SKIP_SPIKE="${SKIP_SPIKE:-0}"

# Examples (replace the custom executable path):
# SPIKE_BIN=/absolute/path/to/spike ./scripts/run_resnet18_benchmarks_selective.sh x1 x2
# SPIKE_BIN=/absolute/path/to/spike ./scripts/run_resnet18_benchmarks_selective.sh "b0,x1"
# SPIKE_BIN=/absolute/path/to/spike MODES="x1 x2" ./scripts/run_resnet18_benchmarks_selective.sh
if (($# > 0)); then
    MODES_SPEC="$*"
else
    MODES_SPEC="${MODES:-b0 b1 x1 x2}"
fi
MODES_SPEC="${MODES_SPEC//,/ }"
read -r -a MODES_ARRAY <<< "$MODES_SPEC"

case "$SKIP_SPIKE" in
    0|1) ;;
    *) echo "SKIP_SPIKE must be 0 or 1" >&2; exit 2 ;;
esac
if (("${#MODES_ARRAY[@]}" == 0)); then
    echo "No benchmark modes specified" >&2
    exit 2
fi

declare -A SEEN_MODES=()
for mode in "${MODES_ARRAY[@]}"; do
    case "$mode" in
        b0|b1|x1|x2) ;;
        *) echo "Invalid mode '$mode'; use b0, b1, x1, x2" >&2; exit 2 ;;
    esac
    if [[ -n "${SEEN_MODES[$mode]:-}" ]]; then
        echo "Duplicate mode '$mode'" >&2
        exit 2
    fi
    SEEN_MODES[$mode]=1
done

cd "$ROOT_DIR"
export PATH="${HOME}/riscv/bin:${PATH}"
if [[ "$SKIP_SPIKE" == 0 ]]; then
    if ! SPIKE_BIN="$(command -v "$SPIKE_BIN")" || [[ ! -x "$SPIKE_BIN" ]]; then
        echo "Specify SPIKE_BIN as the path to your executable custom Spike build" >&2
        exit 2
    fi
    command -v timeout >/dev/null
fi
mkdir -p build results/raw results/tables
run_dir="$(mktemp -d "$ROOT_DIR/build/resnet18_benchmark.XXXXXX")"
trap 'status=$?; if ((status != 0)); then echo "Run failed; unpublished artifacts: $run_dir" >&2; fi' EXIT
echo "Modes: ${MODES_ARRAY[*]}; SKIP_SPIKE=$SKIP_SPIKE"

# Stage every selected mode. Do not mix fresh disassembly with old histograms,
# or publish a new summary if any selected mode fails.
for mode in "${MODES_ARRAY[@]}"; do
    prefix="$run_dir/resnet18_${mode}"
    elf="$ROOT_DIR/build/resnet18_${mode}.elf"
    ./scripts/build_resnet18_b0.sh "$mode" >"$prefix.build.txt"
    elf_hash="$(sha256sum "$elf")"
    riscv64-unknown-elf-size -A "$elf" >"$prefix.size.txt"
    riscv64-unknown-elf-objdump -d "$elf" >"$prefix.disassembly.txt"

    if [[ "$SKIP_SPIKE" == 1 ]]; then
        continue
    fi

    set +e
    timeout "$SPIKE_TIMEOUT" "$SPIKE_BIN" --isa=RV64IMC -g "$elf" \
        >"$prefix.stdout" 2>"$prefix.histogram"
    status=$?
    set -e
    if ((status != 0)); then
        echo "Spike ${mode^^} failed with status $status" >&2
        exit "$status"
    fi
    if [[ "$elf_hash" != "$(sha256sum "$elf")" ]]; then
        echo "ELF changed during Spike execution: $elf" >&2
        exit 1
    fi

    python3 scripts/analyze_histogram.py \
        "$prefix.histogram" "$prefix.disassembly.txt" "$prefix.csv" \
        --mode "$mode"

    # Record the actual executable and paired artifact hashes for this run.
    python3 - "$mode" "$elf" "$SPIKE_BIN" "$prefix" <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

mode, elf, spike, prefix = sys.argv[1:]
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest = {
    "mode": mode,
    "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "spike_bin": str(Path(spike).resolve()),
    "spike_args": ["--isa=RV64IMC", "-g"],
    "compiler": subprocess.check_output(
        ["riscv64-unknown-elf-gcc", "--version"], text=True
    ).splitlines()[0],
    "elf_sha256": sha256(elf),
    "spike_sha256": sha256(spike),
    "histogram_sha256": sha256(prefix + ".histogram"),
    "disassembly_sha256": sha256(prefix + ".disassembly.txt"),
    "executor_sha256": sha256("src/benchmarks/resnet18_b0_executor.c"),
    "analyzer_sha256": sha256("scripts/analyze_histogram.py"),
    "graph_sha256": sha256("model/source/resnet18/graph_report.json"),
}
Path(prefix + ".run.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
done

if [[ "$SKIP_SPIKE" == 0 ]]; then
    python3 scripts/summarize_resnet18.py "${MODES_ARRAY[@]}" \
        --raw-dir "$run_dir" --tables-dir "$run_dir" \
        --output "$run_dir/resnet18_summary.csv"
fi

for mode in "${MODES_ARRAY[@]}"; do
    prefix="$run_dir/resnet18_${mode}"
    if [[ "$SKIP_SPIKE" == 1 ]]; then
        # Keep existing measured artifact pairs and CSVs untouched.
        for suffix in build.txt size.txt disassembly.txt; do
            mv "$prefix.$suffix" "results/raw/resnet18_${mode}.build_only.$suffix"
        done
    else
        for suffix in build.txt size.txt disassembly.txt stdout histogram run.json; do
            mv "$prefix.$suffix" "results/raw/resnet18_${mode}.$suffix"
        done
        mv "$prefix.csv" "results/tables/resnet18_${mode}.csv"
    fi
done
if [[ "$SKIP_SPIKE" == 0 ]]; then
    # All rows come from this invocation, never from unselected old CSVs.
    mv "$run_dir/resnet18_summary.csv" results/tables/resnet18_summary.csv
    echo "Summary: results/tables/resnet18_summary.csv (selected modes only)"
else
    echo "Build-only outputs: results/raw/resnet18_<mode>.build_only.*"
    echo "Measured histograms, disassemblies, and CSVs were not changed."
fi
rmdir "$run_dir"
