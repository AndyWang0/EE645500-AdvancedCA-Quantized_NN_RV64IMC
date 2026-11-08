#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
OUT_DIR="${1:-build/resnet18_b0_tensors}"
mkdir -p "$OUT_DIR"

python3 scripts/make_input_header.py \
    model/test_vectors/resnet18/input_0.bin build/resnet18_input.h
cc -std=c11 -O3 -Wall -Wextra -Werror \
    -I build -I model/exported/resnet18 \
    src/benchmarks/resnet18_b0_dump_main.c \
    src/benchmarks/resnet18_b0_executor.c \
    src/kernels/qgemm.c \
    -o build/resnet18_b0_host_dump
build/resnet18_b0_host_dump "$OUT_DIR"

python3 - "$OUT_DIR" <<'PY'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
source = pathlib.Path("model/test_vectors/resnet18/validation_metadata.json")
metadata = json.loads(source.read_text())
for tensor in metadata["intermediates"]:
    index = int(tensor["index"])
    tensor["available"] = (out / f"tensor_{index}.bin").exists()
    tensor["file"] = f"tensor_{index}.bin"
metadata["intermediate_tensor_count"] = sum(
    tensor["available"] for tensor in metadata["intermediates"]
)
(out / "validation_metadata.json").write_text(
    json.dumps(metadata, indent=2) + "\n"
)
PY

echo "B0 host tensor dump written to $OUT_DIR"
