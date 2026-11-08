#!/usr/bin/env python3
"""Analyze a single-hart Spike PC histogram against its matching disassembly."""

import argparse
import collections
import csv
import json
import math
from pathlib import Path
import re
import sys

HISTOGRAM_RE = re.compile(r"^([0-9a-fA-F]+)\s+([0-9]+)$")
HEADER_RE = re.compile(r"^PC Histogram size:\s*([0-9]+)$")
INSTRUCTION_RE = re.compile(
    r"^\s*([0-9a-fA-F]+):\s+([0-9a-fA-F]+)(?:\s+([a-zA-Z0-9_.]+))?"
)
SYMBOL_RE = re.compile(r"^\s*([0-9a-fA-F]+)\s+<([^>]+)>:")
DEFAULT_GRAPH = (
    Path(__file__).resolve().parents[1] /
    "model/source/resnet18/graph_report.json"
)


def classify(mnemonic):
    if mnemonic in {"xqmac8", "xqrequant8"}:
        return "custom"
    name = mnemonic.removeprefix("c.")
    if name in {"lb", "lh", "lw", "ld", "lbu", "lhu", "lwu",
                "lwsp", "ldsp"}:
        return "load"
    if name in {"sb", "sh", "sw", "sd", "swsp", "sdsp"}:
        return "store"
    if name in {"beq", "bne", "blt", "bge", "bltu", "bgeu",
                "beqz", "bnez", "blez", "bgez", "bltz", "bgtz",
                "bgt", "ble", "bgtu", "bleu",
                "j", "jr", "jal", "jalr", "ret", "call", "tail"}:
        return "branch"
    if name in {"mul", "mulh", "mulhsu", "mulhu", "mulw"}:
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
    if not instructions:
        raise ValueError("disassembly contains no instructions")
    return instructions, symbols


def read_histogram(path):
    counts = {}
    size = None
    with open(path, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            header = HEADER_RE.fullmatch(line.strip())
            if header:
                if size is not None:
                    raise ValueError("multiple histograms: expected one hart/run")
                size = int(header.group(1))
            match = HISTOGRAM_RE.fullmatch(line.strip())
            if match:
                pc, count = int(match.group(1), 16), int(match.group(2))
                if pc in counts:
                    raise ValueError(f"duplicate histogram PC: 0x{pc:x}")
                if count <= 0:
                    raise ValueError(f"nonpositive histogram count: 0x{pc:x}")
                counts[pc] = count
    if not counts or size != len(counts):
        raise ValueError(
            f"incomplete histogram: header={size}, parsed PCs={len(counts)}"
        )
    return counts


def check_coverage(counts, instructions):
    """Only Spike's default reset ROM is deliberately excluded from ELF scope."""
    excluded = 0
    for pc, count in counts.items():
        if pc not in instructions:
            if 0x1000 <= pc < 0x2000:
                excluded += count
            else:
                raise ValueError(
                    f"executed PC 0x{pc:x} absent from disassembly; "
                    "histogram/ELF may be mismatched"
                )
    return excluded


def summarize(counts, instructions, start=None, end=None):
    categories = collections.Counter()
    custom = collections.Counter()
    total = 0
    for pc, count in counts.items():
        if start is not None and not (start <= pc < end):
            continue
        mnemonic = instructions.get(pc)
        if mnemonic is None:
            continue  # check_coverage must run before accepting a report.
        total += count
        categories[classify(mnemonic)] += count
        if mnemonic in {"xqmac8", "xqrequant8"}:
            custom[mnemonic] += count
    return total, categories, custom


def function_ranges(name, symbols, instructions):
    """Include compiler clones; report self PCs, never infer call-tree totals."""
    limit = max(instructions) + 4
    ranges = []
    for symbol, start in symbols.items():
        if symbol == name or symbol.startswith(name + "."):
            end = min(
                (address for address in symbols.values() if address > start),
                default=limit,
            )
            if (start, end) not in ranges:
                ranges.append((start, end))
    return ranges


def workload_metrics(graph_path):
    report = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    conv_macs = conv_outputs = packed_macs = fc_macs = fc_outputs = 0
    for op in report["operators"]:
        if op["operator"] not in {"CONV_2D", "FULLY_CONNECTED"}:
            continue
        weights = op["input_tensors"][1]["shape"]
        outputs = math.prod(op["output_tensors"][0]["shape"])
        reduction = math.prod(weights[1:])
        if op["operator"] == "CONV_2D":
            conv_macs += outputs * reduction
            conv_outputs += outputs
            packed_macs += outputs * (reduction // 8)
        else:
            fc_macs += outputs * reduction
            fc_outputs += outputs
    return {
        "complete_model": (conv_macs + fc_macs,
                           math.prod(report["outputs"][0]["shape"])),
        "convolution": (conv_macs, conv_outputs),
        "fully_connected": (fc_macs, fc_outputs),
        "packed_macs": packed_macs,
    }


def check_mode(mode, custom, metrics):
    mac, requant = custom["xqmac8"], custom["xqrequant8"]
    if mode in {"b0", "b1"} and (mac or requant):
        raise ValueError(f"{mode.upper()} executed custom instructions")
    if mode in {"x1", "x2"} and mac != metrics["packed_macs"]:
        raise ValueError(
            f"{mode.upper()} xqmac8 count {mac} != expected "
            f"{metrics['packed_macs']} for the frozen graph"
        )
    if mode == "x1" and requant:
        raise ValueError("X1 unexpectedly executed xqrequant8")
    if mode == "x2" and requant == 0:
        raise ValueError("X2 did not execute xqrequant8")


def analyze(histogram, disassembly, output, graph=DEFAULT_GRAPH, mode=None):
    counts = read_histogram(histogram)
    instructions, symbols = read_disassembly(disassembly)
    excluded = check_coverage(counts, instructions)
    start, end = symbols.get("benchmark_start"), symbols.get("benchmark_end")
    if (start is None or end is None or start >= end or
            counts.get(start) != 1 or counts.get(end) != 1):
        raise ValueError("expected one completed benchmark_start/end invocation")
    metrics = workload_metrics(graph)
    total, categories, custom = summarize(counts, instructions)
    if mode:
        check_mode(mode, custom, metrics)
    rows = [("complete_model", "elf_text", total, categories, custom,
             *metrics["complete_model"])]
    for region, name, workload in (
        ("convolution_kernel", "conv2d_scalar", "convolution"),
        ("b1_convolution_kernel", "conv2d_b1", "convolution"),
        ("custom_convolution_kernel", "conv2d_custom", "convolution"),
        ("fully_connected_kernel", "fully_connected_scalar", "fully_connected"),
    ):
        ranges = function_ranges(name, symbols, instructions)
        if not ranges:
            continue
        total = 0
        categories, custom = collections.Counter(), collections.Counter()
        for begin, finish in ranges:
            subtotal, subcategories, subcustom = summarize(
                counts, instructions, begin, finish
            )
            total += subtotal
            categories.update(subcategories)
            custom.update(subcustom)
        if mode == "b1" and name == "conv2d_b1":
            if any(instructions[pc] in {"xqmac8", "xqrequant8"}
                   for begin, finish in ranges
                   for pc in instructions if begin <= pc < finish):
                raise ValueError("B1 convolution contains a custom opcode")
        macs, outputs = metrics[workload] if total else (0, 0)
        rows.append((region, "function_self", total, categories, custom,
                     macs, outputs))

    with open(output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "region", "counting_scope", "dynamic_instructions", "loads",
            "stores", "integer", "multiply", "branches", "custom", "xqmac8",
            "xqrequant8", "logical_macs", "outputs", "text_bytes",
        ])
        for name, scope, total, categories, custom, macs, outputs in rows:
            writer.writerow([
                name, scope, total, categories["load"], categories["store"],
                categories["integer"], categories["multiply"],
                categories["branch"], categories["custom"],
                custom["xqmac8"], custom["xqrequant8"], macs, outputs, "",
            ])
    print(
        f"{histogram}: {rows[0][2]} ELF-text instructions; "
        f"excluded {excluded} reset-ROM instructions",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("histogram")
    parser.add_argument("disassembly")
    parser.add_argument("output")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--mode", choices=("b0", "b1", "x1", "x2"))
    args = parser.parse_args()
    try:
        analyze(args.histogram, args.disassembly, args.output,
                args.graph, args.mode)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"analyze_histogram.py: {error}\n")


if __name__ == "__main__":
    main()
