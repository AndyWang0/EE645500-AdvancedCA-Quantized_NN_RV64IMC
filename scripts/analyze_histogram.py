#!/usr/bin/env python3
"""Analyze Spike's bounded PC histogram without retaining an instruction trace."""

import collections
import csv
import re
import sys

HISTOGRAM_RE = re.compile(r"^([0-9a-fA-F]+)\s+([0-9]+)$")
INSTRUCTION_RE = re.compile(
    r"^\s*([0-9a-fA-F]+):\s+([0-9a-fA-F]+)(?:\s+([a-zA-Z0-9_.]+))?"
)
SYMBOL_RE = re.compile(r"^\s*([0-9a-fA-F]+)\s+<([^>]+)>:")


def classify(mnemonic):
    if mnemonic in {"xqmac8", "xqrequant8"}:
        return "custom"
    if mnemonic.startswith(("lb", "lh", "lw", "ld", "lbu", "lhu", "lwu")):
        return "load"
    if mnemonic.startswith(("sb", "sh", "sw", "sd")):
        return "store"
    if mnemonic.startswith(("b", "c.beq", "c.bne", "jal", "c.j", "jalr", "c.jr")):
        return "branch"
    if mnemonic in {"mul", "mulh", "mulhsu", "mulhu", "mulw"}:
        return "multiply"
    return "integer"


def read_disassembly(path):
    instructions = {}
    symbols = {}
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            symbol = SYMBOL_RE.match(line)
            if symbol:
                symbols[symbol.group(2)] = int(symbol.group(1), 16)
            instruction = INSTRUCTION_RE.match(line)
            if instruction:
                word = int(instruction.group(2), 16)
                mnemonic = instruction.group(3) or ""
                if word & 0xFE00707F == 0x0000000B:
                    mnemonic = "xqmac8"
                elif word & 0xFE00707F == 0x0200000B:
                    mnemonic = "xqrequant8"
                instructions[int(instruction.group(1), 16)] = mnemonic
    return instructions, symbols


def read_histogram(path):
    counts = {}
    with open(path, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = HISTOGRAM_RE.match(line.strip())
            if match:
                counts[int(match.group(1), 16)] = int(match.group(2))
    return counts


def summarize(counts, instructions, start=None, end=None):
    categories = collections.Counter()
    custom = collections.Counter()
    total = 0
    for pc, count in counts.items():
        if start is not None and not (start <= pc < end):
            continue
        mnemonic = instructions.get(pc)
        if mnemonic is None:
            continue
        total += count
        categories[classify(mnemonic)] += count
        if mnemonic in {"xqmac8", "xqrequant8"}:
            custom[mnemonic] += count
    return total, categories, custom


def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            f"Usage: {sys.argv[0]} HISTOGRAM DISASSEMBLY OUTPUT_CSV"
        )
    histogram, disassembly, output = sys.argv[1:]
    counts = read_histogram(histogram)
    instructions, symbols = read_disassembly(disassembly)
    rows = []
    total, categories, custom = summarize(counts, instructions)
    rows.append(("complete_model", total, categories, custom))
    function_ranges = (
        ("convolution_kernel", "conv2d_scalar", "conv2d_scalar"),
        ("b1_convolution_kernel", "conv2d_b1", "conv2d_b1"),
        ("custom_convolution_kernel", "conv2d_custom", "conv2d_custom"),
        ("fully_connected_kernel", "fully_connected_scalar", "fully_connected_scalar"),
    )
    for name, start_name, end_name in function_ranges:
        if start_name not in symbols:
            continue
        start = symbols[start_name]
        candidates = [address for address in symbols.values() if address > start]
        end = min(candidates) if candidates else start
        total, categories, custom = summarize(counts, instructions, start, end)
        rows.append((name, total, categories, custom))

    with open(output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "region", "dynamic_instructions", "loads", "stores", "integer",
            "multiply", "branches", "custom", "xqmac8", "xqrequant8",
            "logical_macs", "outputs", "text_bytes",
        ])
        for name, total, categories, custom in rows:
            writer.writerow([
                name,
                total,
                categories["load"],
                categories["store"],
                categories["integer"],
                categories["multiply"],
                categories["branch"],
                categories["custom"],
                custom["xqmac8"],
                custom["xqrequant8"],
                1814073344,
                1000,
                "",
            ])


if __name__ == "__main__":
    main()
