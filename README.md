# Quantized Neural-Network Acceleration on RV64IMC

Bare-metal W8A8 ResNet-18 inference with two custom RV64 instructions:
`xqmac8` performs eight signed INT8 multiply-accumulates; `xqrequant8`
performs fixed-point scaling, rounding, zero-point addition, and saturation.

The midterm evaluation covers ISA semantics, functional verification, and
dynamic instruction counts in modified Spike. CV-Wally RTL, cycle/CPI,
frequency, area, and PPA measurements are outside this evaluation.

## Midterm deliverables

| Component | Location |
|---|---|
| Bare-metal workload and kernels | [Executor](src/benchmarks/resnet18_b0_executor.c), [kernels](src/kernels/) |
| Quantized model and conversion | [Model provenance](model/source/resnet18/model_metadata.json), [exporter](scripts/export_litert_model.py), [model workflow](docs/port_model.md) |
| Custom ISA and Spike integration | [Design](DESIGN.md), [instruction bodies and integration guide](src/isa/) |
| Verification tests | [Tests](tests/), [instruction runner](scripts/test_custom_spike.sh) |
| Reproducible benchmarks | [ResNet-18 runner](scripts/run_resnet18_benchmarks.sh), [synthetic runner](scripts/run_synthetic_benchmarks.sh) |
| Presentation | [Slides](https://docs.google.com/presentation/d/1YkspZg766Fy4EwkhXZw8v3ISbdjjlEsFkUXAtssOfNY/edit?usp=sharing) |
| Methodology and measurements | [Experiments](EXPERIMENTS.md) |
| AI assistance disclosure | [AI usage](AI_USAGE.md) |


## Results

The frozen graph contains 37 operators and 1,814,073,344 logical MACs.
All four modes passed tensor-level validation for the deterministic fixture.

| Mode | Convolution | Dynamic instructions | `xqmac8` | `xqrequant8` | MAC/instruction | `.text` bytes |
|---|---|---:|---:|---:|---:|---:|
| B0 | Scalar | 25,847,519,995 | 0 | 0 | 0.070184 | 9,178 |
| B1 | Software-optimized scalar | 8,987,379,995 | 0 | 0 | 0.201847 | 11,234 |
| X1 | B1 traversal with packed MAC | 8,609,609,995 | 226,394,112 | 0 | 0.210703 | 11,436 |
| X2 | X1 with custom requantization | 8,504,784,995 | 226,394,112 | 2,384,360 | 0.213300 | 11,526 |

Source: [complete-model summary](results/tables/resnet18_summary.csv).
Counts cover executed ELF instructions, including setup, checksum, and
termination; reset-ROM instructions are excluded. B0/B1 execute no custom
instructions. Incremental reductions are 65.2292% for B0 → B1, 4.2033% for
B1 → X1, and 1.2175% for X1 → X2. Runtime weight packing limits the ISA gain.

The [synthetic GEMM](results/tables/synthetic/summary.csv) uses M=8, N=16,
K=32 and retires 62,003 / 54,053 / 39,537 instructions in B0 / X1 / X2.
Its half-open kernel window differs from the complete-model scope.
These measurements establish instruction-count reductions, not cycle speedups.
Operator contributions, bottlenecks, and target assessment are in
[EXPERIMENTS.md](EXPERIMENTS.md).

## Requirements

- Bash, Python 3.10 or compatible, GCC, Git, curl, and sha256sum.
- RISC-V GCC 10.2.0 or compatible, with `riscv64-unknown-elf-objdump`
  and `riscv64-unknown-elf-size`.
- Host LiteRT dependencies from the pinned
  [requirements](model/source/resnet18/host_requirements.txt).
- An external custom Spike build implementing both instructions.

Keep Spike outside this repository. Follow the
[integration guide](src/isa/spike_custom_instruction.md) to register the
encodings, install the instruction bodies, and build the simulator.
The recorded custom revision is
`a80897fc5b7b6c491019b58145338e1678a30366`; it is local provenance, not
an upstream checkout target. 

Every simulator command below requires the path to the custom executable;
replace `/absolute/path/to/riscv-isa-sim/build/spike`. An unmodified system
Spike cannot execute these instructions.

| Variable | Meaning |
|---|---|
| `SPIKE_BIN` | Custom executable; used by all Spike runners |
| `SPIKE_DIR` | Synthetic-runner fallback checkout containing `build/spike`, used only if `SPIKE_BIN` is unset |
| `SPIKE_SOURCE` | External source checkout for model-freeze provenance |

All commands run from the repository root. Target programs use static storage
and HTIF (Host-Target Interface) termination without an operating system or `pk`.

## 1. Prepare the model

Create the host environment:

~~~sh
python3 -m venv .venv
.venv/bin/python -m pip install -r model/source/resnet18/host_requirements.txt
~~~

Acquire the frozen model and its upstream model card:

~~~sh
# Specify the custom Spike executable and source checkout.
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" \
SPIKE_SOURCE="/absolute/path/to/riscv-isa-sim" ./scripts/freeze_resnet18.sh
~~~

The selected model is `litert-community/resnet18/resnet18_int8_channelwise.tflite`,
with SHA-256:

~~~text
8fea1acb62829e120b4953b6fbf0f842e0032909254f51c2d989703b66534683
~~~

The freeze script verifies both downloads and records provenance in
[model_metadata.json](model/source/resnet18/model_metadata.json).
The large model and generated ResNet headers are ignored by Git.

Inspect, generate the host reference, and export static C data:

~~~sh
PYTHONPATH="$PWD" .venv/bin/python scripts/inspect_litert_model.py \
    --model model/source/resnet18/resnet18_int8_channelwise.tflite \
    --json model/source/resnet18/graph_report.json \
    --summary model/source/resnet18/graph_summary.txt

PYTHONPATH="$PWD" .venv/bin/python scripts/validate_litert_model.py \
    --model model/source/resnet18/resnet18_int8_channelwise.tflite \
    --out-dir model/test_vectors/resnet18 \
    --debug-dir model/test_vectors/host_validation/intermediate

PYTHONPATH="$PWD" .venv/bin/python scripts/export_litert_model.py \
    --model model/source/resnet18/resnet18_int8_channelwise.tflite \
    --out-dir model/exported/resnet18 \
    --test-dir model/test_vectors/resnet18
~~~

The deterministic input seed is `20261104`. The
[model workflow](docs/port_model.md) describes layouts, quantization, generated
artifacts, and numerical limitations.

## 2. Verify correctness

Host kernel and runner regressions:

~~~sh
mkdir -p build
gcc -std=c11 -Wall -Wextra -Werror -O3 \
    tests/test_qgemm.c src/kernels/qgemm.c -o build/test_qgemm
./build/test_qgemm
python3 -m unittest discover -s tests -p 'test_*.py' -v
~~~

Expected host result:
`qgemm scalar/custom comparison passed (128 outputs)`.
The Python regression suite contains 19 tests.

Instruction and synthetic GEMM tests:

~~~sh
python3 scripts/export_synthetic_model.py
# Specify the custom Spike executable in each invocation.
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/test_custom_spike.sh
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/test_synthetic_qgemm.sh
~~~

Expected custom counts are 67 MAC / 73 requant operations for the instruction
test, 512 / 0 for synthetic X1, and 512 / 128 for synthetic X2.
See the [synthetic workflow](docs/synthetic_workflow.md) for completion checks.

Full-model validation:

~~~sh
# Specify the custom Spike executable for each mode.
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/validate_resnet18_b0_spike.sh b0
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/validate_resnet18_b0_spike.sh b1
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/validate_resnet18_b0_spike.sh x1
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/validate_resnet18_b0_spike.sh x2
~~~

Each mode should report `Spike <MODE> validation passed`. Validation compares
available nonconstant tensors and reports the first tensor mismatch, element
index, and expected/actual bytes. Keep `SPIKE_INSTRUCTIONS=0` for complete
validation; instruction-budget exhaustion is not proof of completion.

## 3. Build and benchmark

Build-only commands do not invoke Spike:

~~~sh
./scripts/build_resnet18_b0.sh b0
./scripts/build_resnet18_b0.sh b1
./scripts/build_resnet18_b0.sh x1
./scripts/build_resnet18_b0.sh x2
~~~

The target flags include `-march=rv64imc -mabi=lp64 -O3 -ffreestanding`,
`-nostdlib`, and `-nostartfiles`. The mode selector
`RESNET18_EXECUTION_MODE=0/1/2/3` corresponds to B0/B1/X1/X2.

Run complete-model or selected-mode measurements:

~~~sh
# Specify the custom Spike executable.
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/run_resnet18_benchmarks.sh
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/run_resnet18_benchmarks_selective.sh b1 x1 x2
~~~

The runner uses PC histograms (`-g`), not full instruction traces. It stages
all selected modes, checks completion, coverage, and mode isolation, and
publishes:

~~~text
results/raw/resnet18_<mode>.{histogram,disassembly.txt,run.json}
results/tables/resnet18_<mode>.csv
results/tables/resnet18_summary.csv
~~~

Summaries contain only selected modes; B0 comparison is `N/A` if B0 is
unselected. `SKIP_SPIKE=1` produces only `.build_only.*` reports and leaves
measured artifacts unchanged. Correctness validation is a separate gate.

For the small synthetic benchmark:

~~~sh
# Specify the custom Spike executable.
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/run_synthetic_benchmarks.sh
~~~

Outputs are isolated under [results/tables/synthetic/](results/tables/synthetic/)
and [results/raw/synthetic/](results/raw/synthetic/). Source and binary
fixtures reside in model-specific `synthetic/` directories.
Compatibility entry points and output locations are documented in the
[synthetic workflow](docs/synthetic_workflow.md).

New raw artifacts are ignored by Git. Required evidence must be included in
the submission or distributed through an accessible archive.
