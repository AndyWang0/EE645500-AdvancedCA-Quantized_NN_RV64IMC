"""Small parser/report tests; no compilation, model execution, or Spike."""

import collections
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


histogram = load_script("analyze_histogram")
summary = load_script("summarize_resnet18")


class HistogramAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "build").mkdir(exist_ok=True)

    def test_control_and_compressed_memory_aliases(self):
        for mnemonic in ("j", "jr", "ret", "jal", "c.jr", "c.bnez"):
            self.assertEqual(histogram.classify(mnemonic), "branch")
        self.assertEqual(histogram.classify("c.ldsp"), "load")
        self.assertEqual(histogram.classify("c.sdsp"), "store")
        self.assertEqual(histogram.classify("bset"), "integer")

    def test_non_rom_missing_pc_is_not_silently_dropped(self):
        self.assertEqual(histogram.check_coverage({0x1000: 6}, {}), 6)
        with self.assertRaisesRegex(ValueError, "mismatched"):
            histogram.check_coverage({0x80000000: 1}, {})

    def test_mode_contract_rejects_custom_in_b1_and_requant_in_x1(self):
        metrics = {"packed_macs": 8}
        with self.assertRaisesRegex(ValueError, "B1 executed"):
            histogram.check_mode("b1", collections.Counter(xqmac8=8), metrics)
        with self.assertRaisesRegex(ValueError, "X1 unexpectedly"):
            histogram.check_mode(
                "x1", collections.Counter(xqmac8=8, xqrequant8=1), metrics
            )
        with self.assertRaisesRegex(ValueError, "X2 did not"):
            histogram.check_mode("x2", collections.Counter(xqmac8=8), metrics)

    def test_incomplete_and_duplicate_histograms_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as directory:
            path = Path(directory) / "histogram"
            path.write_text("PC Histogram size:2\n80000000 1\n")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                histogram.read_histogram(path)
            path.write_text("PC Histogram size:2\n80000000 1\n80000000 2\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                histogram.read_histogram(path)

    def test_custom_words_decode_without_mnemonic_support(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as directory:
            path = Path(directory) / "disassembly"
            path.write_text(
                "80000000 <conv2d_custom>:\n"
                "80000000: 0000000b .word 0x0000000b\n"
                "80000004: 0200000b .word 0x0200000b\n"
            )
            instructions, _ = histogram.read_disassembly(path)
            self.assertEqual(instructions[0x80000000], "xqmac8")
            self.assertEqual(instructions[0x80000004], "xqrequant8")

    def test_function_self_does_not_include_called_helpers(self):
        instructions = {0x80000000: "jal", 0x80000004: "ret",
                        0x80000008: "xqrequant8"}
        symbols = {"conv2d_custom": 0x80000000,
                   "requantize_channel": 0x80000008}
        ranges = histogram.function_ranges("conv2d_custom", symbols, instructions)
        total, _, custom = histogram.summarize(
            {pc: 10 for pc in instructions}, instructions, *ranges[0]
        )
        self.assertEqual(total, 20)
        self.assertEqual(custom["xqrequant8"], 0)

    def test_partial_summary_never_uses_an_unselected_b0(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as directory:
            root = Path(directory)
            row = {
                "region": "complete_model", "counting_scope": "elf_text",
                "dynamic_instructions": 10, "loads": 2, "stores": 1,
                "integer": 4, "multiply": 1, "branches": 1, "custom": 1,
                "xqmac8": 1, "xqrequant8": 0, "logical_macs": 8,
                "outputs": 2, "text_bytes": "",
            }
            with (root / "resnet18_x1.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            (root / "resnet18_b0.csv").write_text("not a valid current result")
            (root / "resnet18_x1.size.txt").write_text(".text 32 0x80000000\n")
            output = root / "summary.csv"
            summary.summarize(["x1"], root, root, output)
            with output.open(newline="") as stream:
                result = next(csv.DictReader(stream))
            self.assertEqual(result["instruction_change_vs_b0"], "N/A")
            self.assertEqual(result["instructions_per_output"], "5.0000")


if __name__ == "__main__":
    unittest.main()
