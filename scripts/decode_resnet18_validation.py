#!/usr/bin/env python3
"""Decode the first validation marker in a Spike instruction log."""

import argparse
import re
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("elf", type=Path)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()

    symbols = []
    output = subprocess.check_output(
        ["riscv64-unknown-elf-nm", "-n", str(args.elf)],
        text=True,
    )
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[2].startswith("resnet18_validation_"):
            symbols.append((int(fields[0], 16), fields[2]))

    marker = re.compile(r"core\s+\d+:\s+0x([0-9a-f]+).*c\.ebreak")
    for line in args.trace.open(encoding="ascii", errors="replace"):
        match = marker.search(line)
        if not match:
            continue
        address = int(match.group(1), 16)
        candidates = [item for item in symbols if item[0] <= address]
        if not candidates:
            print(f"Validation marker at 0x{address:x}")
            return 1
        name = max(candidates)[1]
        if name == "resnet18_validation_pass":
            print("Spike B0 validation passed")
            return 0
        print(f"Spike B0 validation failed at {name}")
        return 1

    print("No validation marker found")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
