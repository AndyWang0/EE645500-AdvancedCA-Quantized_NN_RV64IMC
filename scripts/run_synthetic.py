#!/usr/bin/env python3
"""Build/run small HTIF workloads, publishing only fully successful invocations."""

import argparse
import collections
import contextlib
import csv
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

if __package__:
    from .analyze_histogram import read_disassembly
    from .analyze_trace import analyze, trace_instructions
    from .export_synthetic_model import export_model
else:
    from analyze_histogram import read_disassembly
    from analyze_trace import analyze, trace_instructions
    from export_synthetic_model import export_model

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ["-std=c11", "-march=rv64imc", "-mabi=lp64", "-mcmodel=medany",
         "-O3", "-ffreestanding", "-fno-builtin", "-Wall", "-Wextra", "-Werror",
         "-nostdlib", "-nostartfiles", "-T", "src/benchmarks/linker.ld"]


class RunFailure(Exception):
    def __init__(self, message, status=1):
        super().__init__(message)
        self.status = status


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash(stream)


def _hash(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def run_spike(spike, elf, prefix, token, timeout):
    trace = prefix.with_suffix(".trace")
    # A unique staging directory plus an absent log prevents old traces passing.
    if trace.exists():
        raise RunFailure("refusing to reuse an existing staged trace")
    command = [spike, "--isa=rv64imc", f"--log={trace}", "-l", str(elf)]
    with prefix.with_suffix(".stdout").open("wb") as stdout, \
            prefix.with_suffix(".stderr").open("wb") as stderr:
        try:
            result = subprocess.run(command, stdout=stdout, stderr=stderr,
                                    timeout=timeout, cwd=ROOT, check=False)
        except subprocess.TimeoutExpired as error:
            raise RunFailure(f"Spike timed out after {timeout:g}s", 124) from error
    if result.returncode != 0:
        raise RunFailure(f"Spike failed with status {result.returncode}",
                         result.returncode if result.returncode > 0 else 1)
    lines = prefix.with_suffix(".stdout").read_text().splitlines()
    if lines.count(f"EE645500_PASS {token}") != 1 or any("EE645500_FAIL" in line for line in lines):
        raise RunFailure("missing/unexpected HTIF pass/fail message")
    if not trace.is_file() or not trace.stat().st_size:
        raise RunFailure("missing/empty fresh Spike trace")
    with trace.open("rb") as stream:
        stream.seek(-1, os.SEEK_END)
        if stream.read(1) != b"\n":
            raise RunFailure("truncated final trace line")
    return command


def check_execution(trace, disassembly, expected):
    instructions, symbols = read_disassembly(disassembly)
    exit_pc = symbols.get("workload_exit")
    if exit_pc is None:
        raise RunFailure("ELF is missing its HTIF exit marker")
    counts = collections.Counter()
    exits = 0
    for pc, mnemonic in trace_instructions(trace):
        if pc not in instructions and not 0x1000 <= pc < 0x2000:
            raise RunFailure(f"trace PC 0x{pc:x} absent from paired disassembly")
        counts[mnemonic] += 1
        exits += pc == exit_pc
        if mnemonic in {"ebreak", "c.ebreak"}:
            raise RunFailure("unexpected breakpoint in trace")
    actual = (counts["xqmac8"], counts["xqrequant8"])
    if actual != expected:
        raise RunFailure(f"custom counts {actual}, expected {expected}")
    if exits != 1:
        raise RunFailure("trace is missing a unique HTIF exit marker")


def write_summary(path, rows):
    baseline = next((int(row["dynamic_instructions"]) for row in rows
                     if row["configuration"] == "B0"), None)
    for row in rows:
        instructions = row["dynamic_instructions"]
        row["instructions_per_output"] = f"{instructions / row['outputs']:.4f}"
        row["macs_per_instruction"] = f"{row['logical_macs'] / instructions:.6f}"
        row["instruction_change_vs_b0"] = (
            f"{(baseline - instructions) / baseline:.4%}" if baseline else "N/A")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def verify_fixture(stage):
    """Reject stale generated headers/vectors without rewriting project assets."""
    headers, vectors = stage / "fixture/headers", stage / "fixture/vectors"
    with contextlib.redirect_stdout(io.StringIO()):
        export_model(ROOT / "model/source/synthetic/model_int8.json", headers, vectors)
    for generated, current in ((headers, ROOT / "model/exported/synthetic"),
                               (vectors, ROOT / "model/test_vectors/synthetic")):
        for path in generated.iterdir():
            target = current / path.name
            if not target.is_file() or target.read_bytes() != path.read_bytes():
                raise RunFailure("stale synthetic fixture; run python3 scripts/export_synthetic_model.py")


def execute(args):
    timeout = float(os.environ.get("SPIKE_TIMEOUT", "30"))
    if not math.isfinite(timeout) or timeout <= 0:
        raise RunFailure("SPIKE_TIMEOUT must be a finite positive number", 2)
    if os.environ.get("SPIKE_INSTRUCTIONS", "0") != "0":
        raise RunFailure("instruction budgets are unsupported; use SPIKE_TIMEOUT", 2)
    spike_spec = os.environ.get("SPIKE_BIN") or str(
        Path(os.environ.get("SPIKE_DIR", str(Path.home() / "src/riscv-isa-sim"))) / "build/spike")
    spike = shutil.which(spike_spec)
    if not spike:
        raise RunFailure("Specify SPIKE_BIN as your custom Spike executable (or SPIKE_DIR checkout)", 2)
    modes = args.modes or ["b0", "x1", "x2"]
    if len(set(modes)) != len(modes):
        raise RunFailure("duplicate benchmark modes", 2)
    output = args.output_root.resolve()
    build = output / "build/synthetic"
    build.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="run.", dir=build))
    print(f"Staging: {stage}", flush=True)
    if args.kind != "instruction-test":
        verify_fixture(stage)
    m = n = k = 0
    if args.kind != "instruction-test":
        shape = json.loads((ROOT / "model/source/synthetic/model_int8.json").read_text())["shape"]
        m, n, k = (shape[key] for key in ("M", "N", "K"))
    names = (["instruction_test"] if args.kind == "instruction-test" else
             [f"qgemm_{mode}_test" for mode in ("x1", "x2")] if args.kind == "gemm-test" else modes)
    rows, publications = [], []
    for name in names:
        instruction_test = args.kind == "instruction-test"
        mode = None if instruction_test else name.split("_")[1] if args.kind == "gemm-test" else name
        prefix = stage / name
        elf = stage / f"{name}.elf"
        sources = ["src/benchmarks/startup.S"]
        defines = []
        if instruction_test:
            sources += ["tests/test_custom_spike.c"]
            expected, token = (67, 73), "instruction_test"
        else:
            sources += ["src/benchmarks/synthetic_gemm_main.c", "src/kernels/qgemm.c"]
            defines = [f"-DBENCHMARK_MODE={dict(b0=0, x1=1, x2=2)[mode]}"]
            expected = (0 if mode == "b0" else m * n * (k // 8), m * n if mode == "x2" else 0)
            token = "synthetic_gemm"
        command = ["riscv64-unknown-elf-gcc", *FLAGS, *defines, *sources, "-o", str(elf)]
        with prefix.with_suffix(".build.txt").open("wb") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        for tool, suffix, options in [("objdump", ".disassembly.txt", ["-d"]),
                                      ("size", ".size.txt", ["-A"])]:
            with prefix.with_suffix(suffix).open("wb") as log:
                subprocess.run([f"riscv64-unknown-elf-{tool}", *options, str(elf)],
                               cwd=ROOT, stdout=log, check=True)
        elf_hash = sha256(elf)
        spike_command = run_spike(spike, elf, prefix, token, timeout)
        if sha256(elf) != elf_hash:
            raise RunFailure("ELF changed during execution")
        check_execution(prefix.with_suffix(".trace"), prefix.with_suffix(".disassembly.txt"), expected)
        if not instruction_test:
            text = next(int(line.split()[1]) for line in prefix.with_suffix(".size.txt").read_text().splitlines()
                        if line.split() and line.split()[0] == ".text")
            csv_path = prefix.with_suffix(".csv")
            row = analyze(prefix.with_suffix(".trace"), prefix.with_suffix(".disassembly.txt"),
                          csv_path, m * n * k, m * n, text)
            if (row["xqmac8"], row["xqrequant8"]) != expected:
                raise RunFailure("custom instructions fell outside the measured kernel window")
            rows.append({"configuration": mode.upper(), "counting_scope": "kernel_window", **row})
            if args.kind == "benchmark":
                publications.append((csv_path, output / f"results/tables/synthetic/{name}.csv"))
        files = sorted(stage.glob(f"{name}.*"))
        manifest = {
            "workload": name, "mode": mode, "passed": True,
            "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "spike_bin": str(Path(spike).resolve()), "spike_command": spike_command,
            "timeout_seconds": timeout, "compiler_command": command,
            "expected_custom_counts": dict(xqmac8=expected[0], xqrequant8=expected[1]),
            "artifact_sha256": {path.name: sha256(path) for path in files},
            "fixture_sha256": sha256(ROOT / "model/source/synthetic/model_int8.json") if not instruction_test else None,
            "source_sha256": {source: sha256(ROOT / source) for source in
                              sources + ["src/benchmarks/linker.ld", "src/benchmarks/baremetal_status.h"]},
        }
        prefix.with_suffix(".run.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for path in sorted(stage.glob(f"{name}.*")):
            if path.suffix == ".elf":
                destination = build / path.name
            elif path.suffix == ".csv" and args.kind == "benchmark":
                continue
            else:
                destination = output / "results/raw/synthetic" / path.name
            publications.append((path, destination))
        print(f"{name} passed ({expected[0]} xqmac8, {expected[1]} xqrequant8)", flush=True)
    if args.kind == "benchmark":
        summary = stage / "summary.csv"
        write_summary(summary, rows)
        publications.append((summary, output / "results/tables/synthetic/summary.csv"))
    # No measured artifact is changed until every selected workload has passed.
    for source, destination in publications:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    print("All selected workloads passed; published fresh synthetic artifacts.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("benchmark", "instruction-test", "gemm-test"))
    parser.add_argument("--modes", nargs="+", choices=("b0", "x1", "x2"))
    parser.add_argument("--output-root", type=Path, default=ROOT,
                        help="Isolate build/results under this directory; source stays in the project.")
    args = parser.parse_args()
    if args.modes and args.kind != "benchmark":
        parser.error("--modes is only supported for benchmark")
    os.environ["PATH"] = str(Path.home() / "riscv/bin") + os.pathsep + os.environ.get("PATH", "")
    try:
        execute(args)
    except RunFailure as error:
        print(f"ERROR: {error} (staged artifacts retained)", file=sys.stderr)
        return error.status
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error} (staged artifacts retained)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
