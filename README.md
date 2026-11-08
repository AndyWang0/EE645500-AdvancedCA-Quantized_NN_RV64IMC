# Quantized Neural-Network Acceleration on RV64IMC

This repository implements and evaluates two custom RV64 instructions for
bare-metal INT8 neural-network inference:

- `xqmac8`: eight signed INT8 products accumulated into a 64-bit register.
- `xqrequant8`: fixed-point multiplication, ties-away-from-zero rounding,
  output zero-point addition, and signed INT8 saturation.

The primary workload is a frozen channelwise-quantized ResNet-18 LiteRT model.
The project is an ISA-level evaluation using modified Spike; it does not claim
cycle, CPI, cache, area, frequency, energy, or hardware speedup results.

## Midterm status

The midterm deliverables are complete:

1. C/C++ bare-metal workload [`src/benchmarks/resnet18_b0_executor.c`](src/benchmarks/resnet18_b0_executor.c) and kernels [`src/kernels/qgemm.c`](src/kernels/qgemm.c). 
2. INT8 model [`model/source/resnet18`](model/source/resnet18/model_metadata.json) and Python conversion/export scripts [`scripts/export_litert_model.py`](scripts/export_litert_model.py).
3. Spike custom-instruction bodies [`src/isa/xqmac8.h`](src/isa/xqmac8.h), [`src/isa/xqrequant8.h`](src/isa/xqrequant8.h), and the external Spike integration guide [`src/isa/spike_custom_instruction.md`](src/isa/spike_custom_instruction.md).
4. Verification tests: short tests to test the added instructions [`scripts/test_custom_spike.sh`](scripts/test_custom_spike.sh).
5. Reproducible synthetic and complete-model benchmark scripts [`scripts/run_resnet18_benchmarks.sh`](scripts/run_resnet18_benchmarks.sh).
6. Slides presentation [`link`](https://docs.google.com/presentation/d/1YkspZg766Fy4EwkhXZw8v3ISbdjjlEsFkUXAtssOfNY/edit?usp=sharing).
7. This README with commands and expected measurements.

The midterm does not include PPA or CV-Wally evaluation. Those are outside the
requested midterm scope.

## Results at a glance

The complete ResNet-18 graph contains 1,814,073,344 logical MACs. The
following measurements were collected with Spike PC histograms and are
reproducible with [`scripts/run_resnet18_benchmarks.sh`](scripts/run_resnet18_benchmarks.sh).

| Mode | Description | Dynamic instructions | `xqmac8` | `xqrequant8` | MAC/instruction | `.text` bytes |
|---|---|---:|---:|---:|---:|---:|
| B0 | Scalar convolution and requantization | 34,651,939,995 | 0 | 0 | 0.052351 | 7,040 |
| B1 | Software-optimized scalar convolution | 34,642,099,995 | 0 | 0 | 0.052366 | 7,034 |
| X1 | `xqmac8` plus scalar requantization | 89,576,554,995 | 226,394,112 | 0 | 0.020252 | 7,110 |
| X2 | `xqmac8` plus `xqrequant8` | 89,471,729,995 | 226,394,112 | 2,384,360 | 0.020275 | 7,208 |

X1 and X2 are functionally correct and execute the custom instructions, but
currently retire more instructions than B0. Packing, temporary stores,
address generation, loop/control overhead, reduction tails, and scalar wide
requantization fallback dominate the current packed implementation. This is
the main bottleneck identified by the midterm experiment.

The small synthetic GEMM gives a complementary, bounded measurement:

| Mode | Dynamic instructions | Scalar multiply | `xqmac8` | `xqrequant8` |
|---|---:|---:|---:|---:|
| B0 | 62,067 | 4,352 | 0 | 0 |
| X1 | 80,190 | 256 | 512 | 0 |
| X2 | 74,820 | 0 | 512 | 128 |

These are instruction counts, not execution cycles.

## Repository map

| Path | Purpose |
|---|---|
| [`DESIGN.md`](DESIGN.md) | ISA encoding, semantics, quantization contract, and rationale |
| [`docs/port_model.md`](docs/port_model.md) | Model acquisition, conversion, export, execution, and validation workflow |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Detailed experiment procedure, tables, and bottleneck analysis |
| [`src/kernels/`](src/kernels/) | Scalar kernels, custom wrappers, and host fallbacks |
| [`src/benchmarks/`](src/benchmarks/) | Bare-metal startup, linker, GEMM, validation, and ResNet executors |
| [`src/isa/`](src/isa/) | ISA-side instruction source used by the modified Spike checkout |
| [`src/isa/spike_custom_instruction.md`](src/isa/spike_custom_instruction.md) | Add, register, build, and verify the instructions in an external Spike checkout |
| [`tests/`](tests/) | Host and bare-metal custom-instruction tests |
| [`scripts/`](scripts/) | Model conversion, build, validation, and benchmark automation |
| [`model/source/resnet18/`](model/source/resnet18/) | Frozen LiteRT model and provenance metadata |
| [`model/exported/resnet18/`](model/exported/resnet18/) | Generated static C graph and tensor data |
| [`results/tables/`](results/tables/) | Machine-readable benchmark summaries |

Generated build files and model artifacts may be ignored in a clean clone;
follow the model preparation steps below before building the full workload.

## Requirements

### Host tools

- Bash
- Python 3.10 or newer
- GCC
- `curl`, `sha256sum`, `git`

### RISC-V tools

- `riscv64-unknown-elf-gcc` 10.2.0 or compatible
- `riscv64-unknown-elf-objdump`
- `riscv64-unknown-elf-size`

### Modified Spike

Keep the Spike source and build **outside this repository**. Follow
[`src/isa/spike_custom_instruction.md`](src/isa/spike_custom_instruction.md)
to install the instruction bodies and register their encodings/build entries.
The guide provides the additions to the four Spike files starting from an
upstream checkout; no existing local modifications are required. The recorded
environment uses:

```text
source:     ~/src/riscv-isa-sim
branch:     midterm-custom-isa
revision:   a80897fc5b7b6c491019b58145338e1678a30366
executable: ~/src/riscv-isa-sim/build/spike
```

If you already have that custom branch and a configured build directory,
rebuild it in its separate checkout (replace the example path):

```sh
cd "/absolute/path/to/riscv-isa-sim"
git switch midterm-custom-isa
make -C build -j2
```

The branch/revision above records the original local environment; it is not
supplied by an upstream clone. For a fresh checkout, use the integration guide.
No Spike binary or build tree is included here.

**Specify your custom Spike path in every command that uses Spike.** The
scripts have different path interfaces; an unmodified system `spike` will
not execute these custom instructions:

| Variable | Meaning | Scripts |
|---|---|---|
| `SPIKE_BIN` | Absolute path to the custom executable | Full-model validation and benchmark scripts |
| `SPIKE_DIR` | Absolute path to the external checkout; executable must be `build/spike` beneath it | Custom instruction/GEMM tests and synthetic benchmarks |
| `SPIKE_SOURCE` | Absolute path to the external source Git checkout, for provenance | Model-freezing script (also requires `SPIKE_BIN`) |

For example, replace the executable path before running:

```sh
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/validate_resnet18_b0_spike.sh x2
```

No `pk` proxy kernel is required. All workloads use the repository startup
code, linker script, static memory, and `tohost` termination.

## 1. Prepare and freeze the model

The selected model is `litert-community/resnet18`:

```text
resnet18_int8_channelwise.tflite
SHA-256: 8fea1acb62829e120b4953b6fbf0f842e0032909254f51c2d989703b66534683
```

If the model is not already present, run:

```sh
# Specify both the custom Spike executable and its source checkout.
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" \
SPIKE_SOURCE="/absolute/path/to/riscv-isa-sim" \
    ./scripts/freeze_resnet18.sh
```

The script downloads the model and source README, verifies both SHA-256
hashes, and records tool/model provenance in
[`model/source/resnet18/model_metadata.json`](model/source/resnet18/model_metadata.json).

Inspect the graph:

```sh
python3 scripts/inspect_litert_model.py \
    --model model/source/resnet18/resnet18_int8_channelwise.tflite \
    --json model/source/resnet18/graph_report.json
```

The model is signed INT8 W8A8 with channelwise INT8 weights and asymmetric
INT8 activations. It contains 20 convolutions, one max pool, eight residual
adds, one average pool, and one fully connected layer.

## 2. Validate and export static model data

Run the host LiteRT reference to produce the deterministic input, final output,
and intermediate tensor snapshots:

```sh
python3 scripts/validate_litert_model.py \
    --model model/source/resnet18/resnet18_int8_channelwise.tflite \
    --out-dir model/test_vectors/resnet18 \
    --debug-dir model/test_vectors/resnet18/intermediate
```

Export the graph and constant tensors as C:

```sh
python3 scripts/export_litert_model.py \
    --model model/source/resnet18/resnet18_int8_channelwise.tflite \
    --out-dir model/exported/resnet18 \
    --test-dir model/test_vectors/resnet18
```

The export preserves tensor shapes, quantization parameters, weights, bias,
weight sums, operator metadata, and the static graph. The executor consumes
these generated artifacts without a runtime or dynamic allocation.

## 3. Run correctness tests

### Host scalar regression

```sh
mkdir -p build
gcc -std=c11 -Wall -Wextra -Werror -O3 \
    tests/test_qgemm.c src/kernels/qgemm.c \
    -o build/test_qgemm
./build/test_qgemm
```

Expected output:

```text
qgemm scalar/custom comparison passed (128 outputs)
```

### Custom Spike instruction test

```sh
# Specify the custom Spike checkout; this script uses SPIKE_DIR/build/spike.
SPIKE_DIR="/absolute/path/to/riscv-isa-sim" ./scripts/test_custom_spike.sh
```

Expected output:

```text
custom Spike instruction test passed (67 xqmac8, 73 xqrequant8)
```

The test covers signed packed lanes, negative and large accumulators, modulo
64-bit accumulation, rounding and saturation boundaries, shift values through
63, random deterministic cases, and `rd = rs1` overlap for `xqrequant8`.

### Full ResNet-18 validation

```sh
# Specify the custom Spike executable for each mode.
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/validate_resnet18_b0_spike.sh b0
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/validate_resnet18_b0_spike.sh b1
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/validate_resnet18_b0_spike.sh x1
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/validate_resnet18_b0_spike.sh x2
```

Expected result for each mode:

```text
Spike B0 validation passed
Spike B1 validation passed
Spike X1 validation passed
Spike X2 validation passed
```

The validator compares available intermediate tensors, not only the final
output. A failure reports the tensor ID, mismatch count, first mismatch index,
expected byte, and actual byte.

## 4. Build a mode

Build one complete bare-metal ELF:

These build-only commands do not invoke Spike and need no Spike path.

```sh
./scripts/build_resnet18_b0.sh b0
./scripts/build_resnet18_b0.sh b1
./scripts/build_resnet18_b0.sh x1
./scripts/build_resnet18_b0.sh x2
```

The build uses `-march=rv64imc`, `-mabi=lp64`, freestanding compilation,
`-nostdlib`, `-nostartfiles`, and the project linker script. The mode is
selected with `RESNET18_EXECUTION_MODE`:

| Value | Mode | Convolution | Requantization |
|---:|---|---|---|
| 0 | B0 | scalar | scalar |
| 1 | B1 | software-optimized scalar | scalar |
| 2 | X1 | packed `xqmac8` | scalar |
| 3 | X2 | packed `xqmac8` | `xqrequant8` where representable |

## 5. Reproduce the measurements

Run the complete-model benchmark:

```sh
# Specify the custom Spike executable.
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/run_resnet18_benchmarks.sh
```

The script builds B0/B1/X1/X2, records ELF sizes and disassemblies, runs the
modified Spike with `-g`, analyzes the PC histograms, and writes:

```text
results/raw/resnet18_<mode>.histogram
results/raw/resnet18_<mode>.disassembly.txt
results/tables/resnet18_<mode>.csv
results/tables/resnet18_summary.csv
```

Histogram mode is required for the complete model. An unrestricted `spike -l`
trace previously grew to approximately 145 GB. The analyzer recognizes custom
instruction encodings even when the unmodified GNU disassembler prints them
as raw words.

For the bounded synthetic GEMM comparison:

```sh
# Specify the custom Spike checkout; this script uses SPIKE_DIR/build/spike.
SPIKE_DIR="/absolute/path/to/riscv-isa-sim" ./scripts/run_benchmarks.sh
```

This produces `results/tables/summary.csv`. The synthetic benchmark uses
`M=8`, `N=16`, `K=32` and 4,096 logical MACs.

## ISA summary

### `xqmac8`

```asm
xqmac8 rd, rs1, rs2
```

`rs1` and `rs2` contain eight signed INT8 lanes in bits `[8*i+7:8*i]`.
The instruction computes the eight products, adds them to the old `rd`, and
writes the result modulo 2^64. Encoding:

```text
custom-0, match 0x0000000b, mask 0xfe00707f
```

### `xqrequant8`

```asm
xqrequant8 rd, rs1, rs2
```

`rs1` is a signed 64-bit accumulator. `rs2` packs a signed 32-bit multiplier
in bits `[31:0]`, a shift `S` in `[37:32]`, and an INT8 output zero point in
`[45:38]`. The instruction performs signed fixed-point scaling with
ties-away-from-zero rounding and saturates to `[-128, 127]`. Encoding:

```text
custom-0, match 0x0200000b, mask 0xfe00707f
```

Full encoding and quantization details are in [`DESIGN.md`](DESIGN.md).

## Design limitations and next optimization

The current X1/X2 software packs bytes into registers inside the innermost
convolution loops and often writes temporary packed data. This explains the
higher load/store, integer, branch, and total instruction counts.

The recommended software optimization is to change the data layout and loop
nesting so that eight contiguous activation and weight lanes are already
available in registers or a reusable packed panel. In order:

1. Prepack weights offline into the exact `xqmac8` lane order.
2. Reuse packed activation panels across output channels and neighboring
   output positions.
3. Hoist address generation and weight-sum correction outside the reduction
   loop.
4. Use a specialized fast path for `K % 8 == 0`; isolate tails and wide
   requantization fallback from the common path.
5. Measure each change against B0, B1, X1, and X2 with the same histogram
   protocol.

The current result is therefore an honest ISA evaluation: the extension is
implemented and verified, while the measurement identifies the software
mapping—not the instruction semantics—as the dominant remaining bottleneck.

## Further documentation

- [`EXPERIMENTS.md`](EXPERIMENTS.md): full methodology, tables, and analysis.
- [`docs/port_model.md`](docs/port_model.md): detailed model-porting record.
- [`DESIGN.md`](DESIGN.md): canonical custom ISA specification.
- [`src/isa/spike_custom_instruction.md`](src/isa/spike_custom_instruction.md): external Spike source additions, build instructions, and verification from an upstream checkout.
