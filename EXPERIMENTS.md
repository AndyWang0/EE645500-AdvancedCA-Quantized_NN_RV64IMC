# ResNet-18 ISA Experiments

## Updated user-run snapshot and script audit

The user has now measured the shared patch-based convolution. The audited
whole-program totals are B0 25,847,519,995, B1 8,987,379,995, X1 8,609,609,995,
and X2 8,504,784,995. B1 executes neither custom instruction. See
[the audit](docs/benchmark_audit.md) for the mode checks, packing bottleneck,
and script fixes. No new model execution was performed during that audit.

## Earlier experiment record

B1 and X1/X2 now use the shared patch-based implementation described in
[the regression diagnosis and rewrite](docs/convolution_optimization.md).
No build, validation, or benchmark was run by the agent for that source
change. All correctness statements and tables below refer to older implementations, not
the current source. The tables below also predate the saved CSVs discussed in
that diagnosis. They are retained as history, not as the updated snapshot above.

This document records reproducible measurements for the quantized ResNet-18
port. The four configurations are B0 (scalar), B1 (software-optimized
scalar), X1 (packed `xqmac8` with scalar requantization), and X2 (packed
`xqmac8` plus `xqrequant8`). The exported graph contains 1,814,073,344
logical MACs and produces 1,000 output values.

## Frozen environment and correctness

The model hash and export metadata are recorded in
`model/source/resnet18/model_metadata.json`. Measurements used Python 3.10.12,
RISC-V GCC 10.2.0, and the modified Spike binary
`/home/andy/src/riscv-isa-sim/build/spike`, from commit
`a80897fc5b7b6c491019b58145338e1678a30366` on `midterm-custom-isa`.

Run the correctness gates before measuring:

```sh
mkdir -p build
gcc -std=c11 -Wall -Wextra -Werror -O3 \
    tests/test_qgemm.c src/kernels/qgemm.c -o build/test_qgemm
./build/test_qgemm
./scripts/test_custom_spike.sh
./scripts/validate_resnet18_b0_spike.sh b0
./scripts/validate_resnet18_b0_spike.sh b1
./scripts/validate_resnet18_b0_spike.sh x1
./scripts/validate_resnet18_b0_spike.sh x2
```

All four model modes passed. Validation compares intermediate tensors and
reports tensor ID, mismatch count, first mismatch index, expected byte, and
actual byte on failure.

## Measurement workflow

Run the complete experiment with:

```sh
./scripts/run_resnet18_benchmarks.sh
```

The script builds every mode, records section sizes and disassembly, and runs
Spike with `--isa=RV64IMC -g`. The histogram mode is intentional: an
unbounded `-l` complete-model trace previously grew to 145 GB. The histogram
is classified by `scripts/analyze_histogram.py`, which decodes custom
instructions from their 32-bit encodings because GNU `objdump` prints them as
unknown raw words. Raw inputs are in `results/raw/`; summaries are in
`results/tables/`.

The analyzer also emits noinline convolution and fully-connected kernel rows.
Those rows aggregate all invocations of each kernel and are not a
single-call trace.

## Complete-model results

These are the measured values from `results/tables/resnet18_summary.csv`.
`integer` excludes loads, stores, branches, scalar multiply, and custom
instructions.

| Mode | Dynamic instructions | Loads | Stores | Integer | Scalar multiply | Branches | `xqmac8` | `xqrequant8` | MAC/instruction | Change vs B0 | `.text` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 34,651,939,995 | 3,611,773,811 | 32,633,255 | 20,107,129,334 | 1,828,001,529 | 9,072,402,066 | 0 | 0 | 0.052351 | 0.00% | 7,040 |
| B1 | 34,642,099,995 | 3,604,292,522 | 22,698,427 | 20,114,705,451 | 1,828,001,529 | 9,072,402,066 | 0 | 0 | 0.052366 | 0.0284% | 7,034 |
| X1 | 89,576,554,995 | 8,905,287,854 | 3,635,038,497 | 56,986,742,113 | 7,026,251,313 | 12,796,841,106 | 226,394,112 | 0 | 0.020252 | -158.5037% | 7,110 |
| X2 | 89,471,729,995 | 8,905,287,854 | 3,635,038,497 | 56,905,760,711 | 7,021,482,593 | 12,775,381,868 | 226,394,112 | 2,384,360 | 0.020275 | -158.2012% | 7,208 |

The custom MAC count is the logical MAC count divided by eight. X2 uses fewer
requantization instructions than the number of packed MAC groups because eight
channels require the exact scalar wide fallback for shifts outside the
current instruction range.

X1/X2 currently retire more instructions than B0. This is a meaningful
optimization result, not a correctness failure: packing, address generation,
tail handling, scalar fallback, and surrounding control work dominate the
benefit of the packed instruction. The next optimization target is to reduce
temporary packing and memory traffic.

## Operator contribution

| Mode | Kernel | Dynamic instructions | Loads | Stores | Scalar multiply | Branches | Custom |
|---|---|---:|---:|---:|---:|---:|---:|
| B0 | convolution | 34,344,036,115 | 3,590,742,664 | 29,990,689 | 1,821,012,800 | 9,024,293,112 | 0 |
| B0 | fully connected | 5,141,053 | 1,027,014 | 1,015 | 513,000 | 517,001 | 0 |
| X1 | custom convolution | 89,268,648,948 | 8,884,256,707 | 3,632,395,931 | 7,019,262,584 | 12,748,732,152 | 226,394,112 |
| X2 | custom convolution | 89,268,648,948 | 8,884,256,707 | 3,632,395,931 | 7,019,262,584 | 12,748,732,152 | 228,778,472 |

Convolution dominates execution; the fully connected layer is comparatively
small. The complete per-mode CSVs contain the B1 convolution row as well.

## Bottlenecks and limitations

* B0 retires 1.828 billion scalar multiplies. B1 reduces stores but does not
  materially change total instructions.
* X1/X2 retire about 226 million packed MACs, but also about 8.91 billion
  loads and 3.64 billion stores, exposing the cost of temporary packing.
* X2 retires 2,384,360 `xqrequant8` operations; scalar wide fallback remains
  necessary for unrepresentable shifts.
* `.text` grows modestly from 7,040 bytes (B0) to 7,110 (X1) and 7,208 (X2).

These are ISA-level measurements only. Spike does not establish cycles, CPI,
cache behavior, area, frequency, energy, or hardware speedup; those require
RTL/CV-Wally evaluation.

## Microbenchmark

Run the bounded synthetic comparison separately:

```sh
./scripts/run_benchmarks.sh
```

Its results are stored in `results/tables/summary.csv`. The microbenchmark
isolates packed MAC behavior, while the complete model includes real graph
shapes, tails, residual operations, and requantization fallback. Do not use
an unbounded `spike -l` run for the complete model.
