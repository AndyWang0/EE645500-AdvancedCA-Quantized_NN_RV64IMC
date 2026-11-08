#!/usr/bin/env python3
"""Strict single-hart trace analysis; the kernel window is [start, end)."""

import collections
import csv
import re
import sys

if __package__:
    from .analyze_histogram import classify, read_disassembly
else:
    from analyze_histogram import classify, read_disassembly

TRACE_RE = re.compile(
    r"^core\s+(\d+):\s+0x([0-9a-fA-F]+)\s+"
    r"\(0x([0-9a-fA-F]+)\)\s+([a-zA-Z0-9_.]+)"
)


def trace_instructions(path):
    """Decode custom words independently of Spike's disassembler spelling."""
    with open(path, encoding="utf-8", errors="strict") as stream:
        for line in stream:
            match = TRACE_RE.match(line)
            if not match:
                if line.startswith("core") and not re.match(r"^core\s+0:\s+>>>>\s+\S", line):
                    raise ValueError("malformed/truncated instruction trace line")
                continue
            if int(match[1]) != 0:
                raise ValueError("expected a single-hart trace (hart 0)")
            word, mnemonic = int(match[3], 16), match[4]
            if word & 0xFE00707F == 0x0000000B:
                mnemonic = "xqmac8"
            elif word & 0xFE00707F == 0x0200000B:
                mnemonic = "xqrequant8"
            elif mnemonic in {"xqmac8", "xqrequant8"}:
                raise ValueError("custom mnemonic disagrees with instruction encoding")
            yield int(match[2], 16), mnemonic


def analyze(trace_path, disassembly_path, output_csv, logical_macs=4096,
            outputs=None, text_bytes=None):
    instructions, symbols = read_disassembly(disassembly_path)
    start, end = symbols.get("benchmark_start"), symbols.get("benchmark_end")
    if start is None or end is None or start >= end:
        raise ValueError("benchmark_start/end symbols are missing or invalid")

    counts = collections.Counter()
    state = "before"
    for pc, mnemonic in trace_instructions(trace_path):
        if pc == start:
            if state != "before":
                raise ValueError("repeated benchmark_start marker")
            state = "active"
        if pc == end:
            if state != "active":
                raise ValueError("unexpected/repeated benchmark_end marker")
            state = "after"
        if state == "active":
            if pc not in instructions:
                raise ValueError(f"kernel PC 0x{pc:x} absent from disassembly")
            # Objdump aliases may differ, but custom decoding must agree.
            if (classify(mnemonic) == "custom" or
                    classify(instructions[pc]) == "custom") and mnemonic != instructions[pc]:
                raise ValueError("trace/disassembly custom instruction mismatch")
            counts[mnemonic] += 1
    if state != "after" or not counts:
        raise ValueError("trace does not contain one complete benchmark window")

    categories = collections.Counter()
    for mnemonic, count in counts.items():
        categories[classify(mnemonic)] += count
    row = {
        "dynamic_instructions": sum(counts.values()),
        "loads": categories["load"], "stores": categories["store"],
        "integer": categories["integer"], "multiply": categories["multiply"],
        "branches": categories["branch"], "custom": categories["custom"],
        "xqmac8": counts["xqmac8"], "xqrequant8": counts["xqrequant8"],
        "logical_macs": logical_macs, "outputs": outputs, "text_bytes": text_bytes,
    }
    with open(output_csv, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    return row


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        raise SystemExit(f"Usage: {sys.argv[0]} TRACE DISASSEMBLY OUTPUT_CSV [LOGICAL_MACS]")
    try:
        print(analyze(*sys.argv[1:4], int(sys.argv[4]) if len(sys.argv) == 5 else 4096))
    except (ValueError, OSError) as error:
        raise SystemExit(str(error))
