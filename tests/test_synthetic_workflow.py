"""Focused runner, trace-accounting, fixture, and failure-publication regressions."""

import contextlib
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_trace as trace
from scripts import export_synthetic_model as exporter
from scripts import run_synthetic as runner


DISASSEMBLY = """80000000 <benchmark_start>:
80000000: 00000013 nop
80000004: 00008067 ret
80000008 <benchmark_end>:
80000008: 00000013 nop
8000000c <workload_exit>:
8000000c: 00000013 nop
"""
VALID_TRACE = """core   0: >>>>  benchmark_start
core   0: 0x80000000 (0x00000013) nop
core   0: 0x80000004 (0x00008067) ret
core   0: 0x80000008 (0x00000013) nop
core   0: 0x8000000c (0x00000013) nop
"""

FAKE_SPIKE = '''#!/usr/bin/env python3
import os
from pathlib import Path
import re
import subprocess
import sys
import time
behavior = os.environ["FAKE_SPIKE_BEHAVIOR"]
log = Path(next(arg[6:] for arg in sys.argv if arg.startswith("--log=")))
if behavior == "timeout":
    time.sleep(10)
if behavior == "late_failure":
    if Path(sys.argv[-1]).stem != "b0":
        sys.exit(7)
    dis = subprocess.check_output(["riscv64-unknown-elf-objdump", "-d", sys.argv[-1]], text=True)
    lines = []
    for symbol in ("benchmark_start", "benchmark_end", "workload_exit"):
        pc = re.search(r"([0-9a-f]+) <" + symbol + r">:", dis)[1]
        lines.append("core   0: 0x" + pc + " (0x00000013) nop")
    log.write_text("\\n".join(lines) + "\\n")
    print("EE645500_PASS synthetic_gemm")
    sys.exit(0)
if behavior != "missing_trace":
    log.write_text("" if behavior == "empty_trace" else
                   "core   0: 0x80000000 (0x00000013) nop" if behavior == "truncated" else
                   "core   0: 0x80000000 (0x00000013) nop\\n")
if behavior != "no_pass":
    print("EE645500_PASS instruction_test")
if behavior == "fail_message":
    print("EE645500_FAIL instruction_test")
sys.exit(7 if behavior == "exit7" else 0)
'''


class SyntheticWorkflowTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic_tests.", dir=ROOT / "build")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.trace = self.directory / "run.trace"
        self.disassembly = self.directory / "run.disassembly.txt"
        self.csv = self.directory / "run.csv"
        self.disassembly.write_text(DISASSEMBLY)
        self.trace.write_text(VALID_TRACE)
        self.fake = self.directory / "fake_spike.py"
        self.fake.write_text(FAKE_SPIKE)
        self.fake.chmod(0o755)

    def test_window_is_half_open_and_counts_aliases(self):
        row = trace.analyze(self.trace, self.disassembly, self.csv)
        self.assertEqual(row["dynamic_instructions"], 2)
        self.assertEqual(row["branches"], 1)
        self.assertEqual(trace.classify("c.jr"), "branch")
        self.assertEqual(trace.classify("bset"), "integer")
        self.assertEqual(trace.classify("c.lwsp"), "load")

    def test_incomplete_reversed_repeated_markers_rejected(self):
        lines = VALID_TRACE.splitlines(keepends=True)
        for broken in ("".join(lines[:3]), "".join(lines[3:]),
                       VALID_TRACE + VALID_TRACE, VALID_TRACE + lines[3]):
            with self.subTest(trace=broken):
                self.trace.write_text(broken)
                with self.assertRaises(ValueError):
                    trace.analyze(self.trace, self.disassembly, self.csv)

    def test_raw_custom_words_and_mnemonic_mismatch(self):
        self.trace.write_text("core 0: 0x80000000 (0x0000000b) unknown\n"
                              "core 0: 0x80000004 (0x0200000b) unknown\n")
        self.assertEqual([mnemonic for _, mnemonic in trace.trace_instructions(self.trace)],
                         ["xqmac8", "xqrequant8"])
        self.trace.write_text("core 0: 0x80000000 (0x00000013) xqmac8\n")
        with self.assertRaises(ValueError):
            list(trace.trace_instructions(self.trace))

    def test_unknown_pc_truncated_line_and_extra_hart_rejected(self):
        for broken in (VALID_TRACE.replace("0x80000004", "0x80000100"),
                       VALID_TRACE.replace("(0x00008067) ret", "(0x000080"),
                       VALID_TRACE.replace("core   0:", "core   1:")):
            self.trace.write_text(broken)
            with self.assertRaises(ValueError):
                trace.analyze(self.trace, self.disassembly, self.csv)

    def test_completion_marker_and_exact_counts_required(self):
        runner.check_execution(self.trace, self.disassembly, (0, 0))
        with self.assertRaises(runner.RunFailure):
            runner.check_execution(self.trace, self.disassembly, (1, 0))
        self.trace.write_text(VALID_TRACE.rsplit("core", 1)[0])
        with self.assertRaises(runner.RunFailure):
            runner.check_execution(self.trace, self.disassembly, (0, 0))

    def test_simulator_failure_paths(self):
        for behavior, status in (("exit7", 7), ("timeout", 124), ("no_pass", 1),
                                 ("fail_message", 1), ("missing_trace", 1),
                                 ("empty_trace", 1), ("truncated", 1)):
            with self.subTest(behavior=behavior):
                prefix = self.directory / behavior
                with mock.patch.dict(os.environ, {"FAKE_SPIKE_BEHAVIOR": behavior}):
                    with self.assertRaises(runner.RunFailure) as caught:
                        runner.run_spike(str(self.fake), self.directory / "test.elf", prefix,
                                         "instruction_test", .15 if behavior == "timeout" else 5)
                self.assertEqual(caught.exception.status, status)

    def test_old_staged_trace_is_never_accepted(self):
        with self.assertRaisesRegex(runner.RunFailure, "reuse"):
            runner.run_spike(str(self.fake), self.directory / "test.elf",
                             self.directory / "run", "instruction_test", 5)

    def test_nop_only_success_is_not_completion(self):
        prefix = self.directory / "nop_only"
        with mock.patch.dict(os.environ, {"FAKE_SPIKE_BEHAVIOR": "nop_only"}):
            runner.run_spike(str(self.fake), self.directory / "test.elf", prefix,
                             "instruction_test", 5)
        with self.assertRaises(runner.RunFailure):
            runner.check_execution(prefix.with_suffix(".trace"), self.disassembly, (67, 73))

    @unittest.skipUnless(shutil.which("riscv64-unknown-elf-gcc"), "requires cross compiler")
    def test_late_selected_mode_failure_preserves_all_published_artifacts(self):
        protected = [self.directory / path for path in (
            "results/tables/synthetic/summary.csv", "results/tables/synthetic/b0.csv",
            "results/raw/synthetic/b0.trace", "build/synthetic/b0.elf")]
        for path in protected:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"old-published-artifact\n")
        environment = {**os.environ, "SPIKE_BIN": str(self.fake),
                       "FAKE_SPIKE_BEHAVIOR": "late_failure", "SPIKE_INSTRUCTIONS": "0"}
        result = subprocess.run([sys.executable, str(ROOT / "scripts/run_synthetic.py"),
                                 "benchmark", "--output-root", str(self.directory)],
                                env=environment, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertIn("b0 passed", result.stdout)
        for path in protected:
            self.assertEqual(path.read_bytes(), b"old-published-artifact\n")
        self.assertFalse((self.directory / "results/tables/synthetic/x1.csv").exists())

    def test_summary_never_uses_unselected_stale_baseline(self):
        rows = [{"configuration": "X1", "dynamic_instructions": 10,
                 "outputs": 2, "logical_macs": 8}]
        runner.write_summary(self.csv, rows)
        with self.csv.open() as stream:
            self.assertEqual(next(csv.DictReader(stream))["instruction_change_vs_b0"], "N/A")

    def test_export_round_trip_and_compatibility_entrypoint(self):
        model = ROOT / "model/source/synthetic/model_int8.json"
        headers, vectors = self.directory / "exported", self.directory / "vectors"
        with contextlib.redirect_stdout(io.StringIO()):
            exporter.export_model(model, headers, vectors)
        for path in headers.iterdir():
            self.assertEqual(path.read_bytes(), (ROOT / "model/exported/synthetic" / path.name).read_bytes())
        for path in vectors.iterdir():
            self.assertEqual(path.read_bytes(), (ROOT / "model/test_vectors/synthetic" / path.name).read_bytes())
        from scripts.export_model import export_model
        self.assertIs(export_model, exporter.export_model)

    def test_stale_export_is_rejected_without_rewriting_it(self):
        runner.verify_fixture(self.directory)
        original = exporter.export_model

        def corrupt(model, headers, vectors):
            original(model, headers, vectors)
            (headers / "weights.h").write_text("stale\n")

        with mock.patch.object(runner, "export_model", side_effect=corrupt):
            with self.assertRaisesRegex(runner.RunFailure, "stale synthetic fixture"):
                runner.verify_fixture(self.directory)


if __name__ == "__main__":
    unittest.main()
