#!/usr/bin/env python3
"""Convert preserved INT8 tensor snapshots into a target C header."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    base = args.metadata.parent
    records = metadata["intermediates"]
    with args.output.open("w", encoding="ascii") as output:
        output.write("#ifndef RESNET18_REFERENCE_TENSORS_H\n")
        output.write("#define RESNET18_REFERENCE_TENSORS_H\n")
        output.write("#include <stdint.h>\n\n")
        for record in records:
            if not record.get("available", False):
                continue
            path = Path(record["file"])
            if not path.is_absolute():
                path = path if path.exists() else base / path
            values = path.read_bytes()
            name = f"resnet18_reference_tensor_{record['index']}"
            output.write(f"static const int8_t {name}[{len(values)}] = {{\n")
            for start in range(0, len(values), 16):
                output.write("    " + ", ".join(
                    str(value - 256 if value >= 128 else value)
                    for value in values[start:start + 16]
                ) + ",\n")
            output.write("};\n\n")
        output.write("#endif\n")


if __name__ == "__main__":
    main()
