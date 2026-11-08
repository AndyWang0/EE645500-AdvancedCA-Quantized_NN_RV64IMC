#!/usr/bin/env python3
"""Summarize only the selected, freshly analyzed model modes."""

import argparse
import csv
from pathlib import Path


def summarize(names, raw_dir, tables_dir, output):
    rows = {}
    for name in names:
        with (tables_dir / f"resnet18_{name}.csv").open(newline="") as stream:
            row = next(csv.DictReader(stream))
        if row["region"] != "complete_model":
            raise ValueError(f"{name}: first row is not complete_model")
        instructions = int(row["dynamic_instructions"])
        categories = ("loads", "stores", "integer", "multiply", "branches", "custom")
        if instructions <= 0 or sum(int(row[key]) for key in categories) != instructions:
            raise ValueError(f"{name}: inconsistent instruction categories")
        if int(row["outputs"]) <= 0:
            raise ValueError(f"{name}: invalid output count")
        row.pop("region")
        row["configuration"] = name.upper()
        with (raw_dir / f"resnet18_{name}.size.txt").open() as stream:
            row["text_bytes"] = next(
                (line.split()[1] for line in stream
                 if line.split() and line.split()[0] == ".text"), None
            )
        if row["text_bytes"] is None:
            raise ValueError(f"{name}: .text size is absent")
        row["instructions_per_output"] = f"{instructions / int(row['outputs']):.4f}"
        row["macs_per_instruction"] = f"{int(row['logical_macs']) / instructions:.6f}"
        rows[name] = row
    baseline = int(rows["b0"]["dynamic_instructions"]) if "b0" in rows else None
    for name, row in rows.items():
        instructions = int(row["dynamic_instructions"])
        row["instruction_change_vs_b0"] = (
            f"{(baseline - instructions) / baseline:.4%}"
            if baseline is not None else "N/A"
        )
    fields = ["configuration"] + [key for key in rows[names[0]] if key != "configuration"]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modes", nargs="+", choices=("b0", "b1", "x1", "x2"))
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.modes)) != len(args.modes):
        parser.error("duplicate modes")
    try:
        summarize(args.modes, args.raw_dir, args.tables_dir, args.output)
    except (OSError, ValueError, KeyError, StopIteration) as error:
        parser.exit(1, f"summarize_resnet18.py: {error}\n")


if __name__ == "__main__":
    main()
