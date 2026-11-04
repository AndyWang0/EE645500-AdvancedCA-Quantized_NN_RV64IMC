#!/usr/bin/env python3
"""Count instructions in named ResNet operator marker ranges."""

import collections
import csv
import re
import sys

TRACE_RE = re.compile(
    r"^core\s+\d+:\s+0x([0-9a-fA-F]+)\s+"
    r"\(0x[0-9a-fA-F]+\)\s+([a-zA-Z0-9_.]+)"
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


def symbols(path):
    result = {}
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            match = SYMBOL_RE.match(line)
            if match:
                result[match.group(2)] = int(match.group(1), 16)
    return result


def analyze(trace_path, disassembly_path, output_path):
    symbol_table = symbols(disassembly_path)
    ranges = (
        ("initial_conv", "resnet18_initial_conv_start", "resnet18_initial_conv_end"),
        ("fully_connected", "resnet18_fc_start", "resnet18_fc_end"),
    )
    counts = {name: collections.Counter() for name, _, _ in ranges}
    active = {name: False for name, _, _ in ranges}
    with open(trace_path, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = TRACE_RE.match(line)
            if not match:
                continue
            pc = int(match.group(1), 16)
            mnemonic = match.group(2)
            for name, start_name, end_name in ranges:
                start = symbol_table.get(start_name)
                end = symbol_table.get(end_name)
                if start is None or end is None:
                    raise SystemExit(f"missing marker for {name}")
                if pc == start:
                    active[name] = True
                if active[name]:
                    counts[name][mnemonic] += 1
                if pc == end and active[name]:
                    active[name] = False

    with open(output_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "operator", "dynamic_instructions", "loads", "stores", "integer",
            "multiply", "branches", "custom", "xqmac8", "xqrequant8",
        ])
        for name, _, _ in ranges:
            counter = counts[name]
            categories = collections.Counter()
            for mnemonic, count in counter.items():
                categories[classify(mnemonic)] += count
            writer.writerow([
                name,
                sum(counter.values()),
                categories["load"],
                categories["store"],
                categories["integer"],
                categories["multiply"],
                categories["branch"],
                categories["custom"],
                counter["xqmac8"],
                counter["xqrequant8"],
            ])


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(
            f"Usage: {sys.argv[0]} TRACE DISASSEMBLY OUTPUT_CSV"
        )
    analyze(sys.argv[1], sys.argv[2], sys.argv[3])
