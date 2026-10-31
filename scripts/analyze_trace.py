#!/usr/bin/env python3
import collections
import csv
import re
import sys

TRACE_RE = re.compile(
    r"^core\s+\d+:\s+0x([0-9a-fA-F]+)\s+"
    r"\(0x[0-9a-fA-F]+\)\s+([a-zA-Z0-9_.]+)"
)
SYMBOL_RE = re.compile(r"^\s*([0-9a-fA-F]+)\s+<([^>]+)>:")


def load_symbol_addresses(path):
    symbols = {}
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            match = SYMBOL_RE.match(line)
            if match:
                symbols[match.group(2)] = int(match.group(1), 16)
    return symbols


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


def analyze(trace_path, disassembly_path, output_csv):
    symbols = load_symbol_addresses(disassembly_path)
    start = symbols.get("benchmark_start")
    end = symbols.get("benchmark_end")
    if start is None or end is None or start >= end:
        raise SystemExit("benchmark_start/end symbols are missing or invalid")

    counts = collections.Counter()
    active = False
    saw_start = False
    saw_end = False
    total = 0
    with open(trace_path, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = TRACE_RE.match(line)
            if not match:
                continue
            pc = int(match.group(1), 16)
            if pc == start:
                active = True
                saw_start = True
            if active:
                counts[match.group(2)] += 1
                total += 1
            if pc == end and active:
                active = False
                saw_end = True

    if not saw_start or not saw_end:
        raise SystemExit("trace does not contain complete benchmark markers")

    categories = collections.Counter()
    for mnemonic, count in counts.items():
        categories[classify(mnemonic)] += count

    row = {
        "dynamic_instructions": total,
        "loads": categories["load"],
        "stores": categories["store"],
        "integer": categories["integer"],
        "multiply": categories["multiply"],
        "branches": categories["branch"],
        "custom": categories["custom"],
        "xqmac8": counts["xqmac8"],
        "xqrequant8": counts["xqrequant8"],
        "logical_macs": 8 * 16 * 32,
        "outputs": 8 * 16,
        "text_bytes": None,
    }
    with open(output_csv, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    print(f"{trace_path}: {total} kernel instructions")
    print(f"xqmac8={row['xqmac8']} xqrequant8={row['xqrequant8']}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(
            f"Usage: {sys.argv[0]} TRACE DISASSEMBLY OUTPUT_CSV"
        )
    analyze(sys.argv[1], sys.argv[2], sys.argv[3])
