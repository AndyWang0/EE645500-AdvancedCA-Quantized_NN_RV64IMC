#!/usr/bin/env python3
"""Convert a canonical INT8 input vector into a C header."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    values = args.input.read_bytes()
    with args.output.open("w", encoding="ascii") as output:
        output.write("#ifndef RESNET18_INPUT_H\n#define RESNET18_INPUT_H\n")
        output.write("#include <stdint.h>\n\n")
        output.write(f"static const int8_t resnet18_input[{len(values)}] = {{\n")
        for start in range(0, len(values), 16):
            output.write("    " + ", ".join(str(value - 256 if value >= 128 else value)
                                             for value in values[start:start + 16]))
            output.write(",\n")
        output.write("};\n\n#endif\n")


if __name__ == "__main__":
    main()
