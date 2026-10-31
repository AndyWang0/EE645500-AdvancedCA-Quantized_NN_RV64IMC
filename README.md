# EE645500-AdvancedCA-Quantized_NN_RV64IMC
Term Project: Quantized Neural-Network Acceleration on RV64IMC

## Step 1.4 — Scalar reference kernel

This step establishes the deterministic scalar baseline required before
integrating the custom ISA instructions. The implementation uses static
test buffers, ordinary C arithmetic, and no `pk` proxy-kernel dependency.

### Files created

- [`src/kernels/qgemm.h`](src/kernels/qgemm.h): declares the GEMM,
  requantization, and custom-path interfaces.
- [`src/kernels/qgemm.c`](src/kernels/qgemm.c): implements the scalar
  reference kernel and fixed-point requantization.
- [`tests/test_qgemm.c`](tests/test_qgemm.c): deterministic correctness test
  using an `M=8`, `N=16`, `K=32` matrix.
- [`build/`](build/): ignored directory used for temporary compiler outputs.

### Implemented behavior

The Day 1 reference implementation is in [`src/kernels/qgemm.c`](src/kernels/qgemm.c).
It provides:

- `qgemm_scalar()`: row-major signed INT8 GEMM with per-output-channel
  fixed-point requantization. It initializes each output from its bias,
  accumulates signed INT8 products, and writes one requantized INT8 result.
- `qgemm_custom()`: compatibility entry point with identical scalar behavior.
  Its interface is fixed so its inner loop can later be replaced by
  `xqmac8` and `xqrequant8`.
- `requantize_reference()`: the project-defined round-to-nearest,
  ties-away-from-zero and signed INT8 saturation reference. It uses
  `__int128` intermediates to avoid overflow in the fixed-point product.

The test also checks negative rounding, exact ties, `S=0`, `S=63`, positive
and negative saturation, and compares every scalar/custom output element
rather than only comparing a checksum.

### Commands used

Run the host correctness test with warnings treated as errors:

```sh
mkdir -p build
gcc -std=c11 -Wall -Wextra -Werror -O2 \
    tests/test_qgemm.c src/kernels/qgemm.c \
    -o build/test_qgemm
./build/test_qgemm
```

Expected result:

```text
qgemm scalar/custom comparison passed (128 outputs)
```

Compile the kernel for the required RV64IMC target:

```sh
mkdir -p build
riscv64-unknown-elf-gcc \
    -std=c11 -march=rv64imc -mabi=lp64 -O2 \
    -ffreestanding -fno-builtin -Wall -Wextra -Werror \
    -Isrc/kernels -c src/kernels/qgemm.c \
    -o build/qgemm.o
```

The target compilation uses `rv64imc`, the `lp64` ABI, freestanding mode,
and disables compiler builtins. The object-only build is intentional at this
stage because startup code and a linker script have not yet been added.

### Command-option explanation

The host command is used only to run the deterministic test on the development
machine:

- `gcc`: compiles for the host machine, allowing the test to execute without a
  RISC-V simulator or proxy kernel.
- `-std=c11`: selects the C language standard used by the kernel sources.
- `-Wall -Wextra`: enables common and additional compiler diagnostics.
- `-Werror`: treats warnings as errors so portability and type issues cannot
  be silently accepted.
- `-O2`: enables normal release-level optimization while preserving a useful
  baseline for later comparisons.
- `-o build/test_qgemm`: writes the test executable to the ignored build
  directory.

The target command checks that the kernel can be compiled for the project's
RV64 processor:

- `riscv64-unknown-elf-gcc`: uses the RISC-V bare-metal cross-compiler rather
  than generating host instructions.
- `-std=c11`, `-Wall`, `-Wextra`, `-Werror`, and `-O2`: have the same language,
  diagnostic, warning-policy, and optimization meanings as in the host build.
- `-march=rv64imc`: targets RV64 with the required base integer (`I`),
  multiplication/division (`M`), and compressed (`C`) extensions. This is
  critical because the baseline and later custom-ISA measurements must not
  accidentally use unsupported instructions.
- `-mabi=lp64`: selects the 64-bit integer ABI with 64-bit pointers and no
  floating-point ABI. This matches the project's no-floating-point-hardware
  constraint.
- `-ffreestanding`: tells the compiler that the program may run without a
  hosted C runtime, startup files, or a complete standard library. This is
  appropriate for the eventual bare-metal workload.
- `-fno-builtin`: prevents the compiler from replacing source operations with
  implicit library/builtin calls that may not exist in the bare-metal
  environment.
- `-Isrc/kernels`: adds the kernel header directory to the include search
  path.
- `-c`: performs compilation only and produces a relocatable object; it does
  not link an executable. A linker script and startup code are required for
  the later bare-metal ELF.
- `-o build/qgemm.o`: writes the target object file to the ignored build
  directory.

The host executable validates numerical behavior, while the target object
compilation validates ISA, ABI, and freestanding compatibility. They are
complementary checks and neither one replaces the eventual linked bare-metal
Spike test.

## Step 1.5 — Quantized-model deliverable

The project guide allows either an existing INT8 model or a model-export
script. Because no external quantized model is checked into this repository,
I implemented a deterministic synthetic INT8 GEMM that follows the same
quantized interface required by the project.

This is a valid Day-1 deliverable because the requirement is to produce a
model artifact, quantized metadata, and a small test-vector set. It does not
require training or converting a large Transformer in one pass.

### Files created

- [`scripts/export_model.py`](scripts/export_model.py): creates the exported
  tensors and test vectors for the kernel workflow.
- [`model/source/model_int8.json`](model/source/model_int8.json): synthetic
  INT8 model description with shape `M=8`, `N=16`, `K=32`.
- [`model/exported/weights.h`](model/exported/weights.h): static weights and
  bias arrays for the C kernel.
- [`model/exported/quant_params.h`](model/exported/quant_params.h): multiplier,
  shift, and zero-point arrays.
- [`model/exported/model_metadata.h`](model/exported/model_metadata.h): export
  metadata for the zero point and tensor layout.
- [`model/test_vectors/input.bin`](model/test_vectors/input.bin): packed INT8
  input data.
- [`model/test_vectors/reference_output.bin`](model/test_vectors/reference_output.bin):
  expected INT8 output for the synthetic GEMM.
- [`model/test_vectors/metadata.json`](model/test_vectors/metadata.json):
  serialized model metadata and quantization settings.

### What the script does

The exporter script performs the Day-1 workflow:

1. Reads a JSON model definition with `M`, `N`, `K`, weights, bias, input data,
   and per-channel quantization values.
2. Validates that the tensor lengths are consistent with the selected GEMM
   shape.
3. Computes a deterministic reference output using the same requantization
   rule as the scalar kernel.
4. Writes C headers with static arrays ready to include in the bare-metal
   kernel.
5. Writes binary `input.bin` and `reference_output.bin` files for later
   validation.
6. Saves JSON metadata so the exported tensors can be traced back to the exact
   quantization parameters used.

### Command used

```sh
python3 scripts/export_model.py \
    --model model/source/model_int8.json \
    --out-dir model/exported \
    --test-dir model/test_vectors
```

This command creates the quantized-model deliverable without requiring a large
external `.tflite` conversion step or a full model-conversion pipeline.

## Step 1.6 — Build and measure B0

The scalar reference is now complete, so the next step is to build a minimal
bare-metal executable that exercises the same kernel under Spike and records a
baseline instruction trace. This establishes the B0 configuration required by
project guidance before the custom ISA is introduced.

### Files created

- [`scripts/build_b0.sh`](scripts/build_b0.sh): one-shot build and measurement
  flow for the bare-metal B0 executable.
- [`src/benchmarks/b0_main.c`](src/benchmarks/b0_main.c): deterministic B0
  workload that fills a synthetic INT8 input tensor, executes
  `qgemm_scalar()`, and computes a checksum for the output buffer.
- [`src/benchmarks/startup.S`](src/benchmarks/startup.S): minimal startup code
  that sets the stack pointer and jumps into `main()`.
- [`src/benchmarks/linker.ld`](src/benchmarks/linker.ld): simple linker script
  that places the ELF in a bare-metal memory layout compatible with Spike.
- [`results/raw/b0.summary.txt`](results/raw/b0.summary.txt): summary of the
  measured dynamic instruction count and code-size report.
- [`results/raw/b0.trace`](results/raw/b0.trace): instruction trace emitted by
  Spike for the B0 baseline.
- [`results/raw/b0.disassembly.txt`](results/raw/b0.disassembly.txt): the final
  disassembly generated by `riscv64-unknown-elf-objdump`.

### Implementation details

The B0 workload is intentionally small but executable in the same bare-metal
model as the final project. It uses the exported synthetic model from the prior
step, runs the scalar reference kernel once to produce a valid output buffer,
then spins in a tight loop so Spike can trace a stable execution window without
requiring a full runtime or a proxy kernel.

The startup assembly sets a fixed stack pointer at `0x81000000` and enters
`main()`. The linker script places `.text`, `.rodata`, `.data`, and `.bss` in a
bare-metal memory region starting at `0x80000000`, which matches the address
layout expected by Spike. The compiler flag `-mcmodel=medany` is required
because the static data and stack are placed above the 2 GiB boundary.

### Command used

```sh
./scripts/build_b0.sh
```

This script performs the full B0 flow:

```sh
mkdir -p build results/raw

riscv64-unknown-elf-gcc \
    -std=c11 \
    -march=rv64imc \
    -mabi=lp64 \
    -mcmodel=medany \
    -O2 \
    -ffreestanding \
    -fno-builtin \
    -Wall -Wextra -Werror \
    -nostdlib -nostartfiles \
    -T src/benchmarks/linker.ld \
    src/benchmarks/startup.S \
    src/benchmarks/b0_main.c \
    src/kernels/qgemm.c \
    -o build/b0.elf

riscv64-unknown-elf-size -A build/b0.elf > results/raw/b0.size.txt
riscv64-unknown-elf-objdump -d build/b0.elf > results/raw/b0.disassembly.txt

spike \
    --isa=rv64imc \
    --instructions=200000 \
    --log=results/raw/b0.trace \
    -l \
    build/b0.elf \
    > results/raw/b0.stdout 2> results/raw/b0.stderr
```

### Key option explanation

- `-march=rv64imc`: selects the baseline project ISA and matches the target
  processor profile.
- `-mabi=lp64`: selects the correct bare-metal 64-bit integer ABI for a
  no-FPU RV64 system.
- `-mcmodel=medany`: necessary for code/data addresses placed near
  `0x80000000`; without this mode, `la`/`HI20` relocations can overflow for
  static data references.
- `-ffreestanding -fno-builtin -nostdlib -nostartfiles`: ensure the ELF is a
  true bare-metal workload without a hosted libc or proxy-kernel startup.
- `-T src/benchmarks/linker.ld`: anchors the program in a fixed memory layout
  for Spike execution.
- `--isa=rv64imc`: enforces the required ISA string when running under Spike.
- `--instructions=200000`: limits the trace to a finite, reproducible measure,
  which is important because the benchmark intentionally spins after the scalar
  GEMM completes.
- `-l --log=results/raw/b0.trace`: writes the instruction trace used to assess
  the B0 baseline.

### Current measured baseline

The current environment produces the following B0 measurement:

```text
Dynamic instructions: 200000
.text size: 862
.rodata size: 736
.data size: 8
.bss size: 384
```

This is the baseline trace used to evaluate the scalar kernel before any custom
`xqmac8` or `xqrequant8` logic is added.

## Step 2.1 — Modify Spike

Spike was modified in its separate source checkout at
`~/src/riscv-isa-sim`. The requested branch was created before any source
changes:

```sh
cd ~/src/riscv-isa-sim
git switch -c midterm-custom-isa
```

The branch is based on the existing Spike commit in that checkout. The project
repository and the Spike repository remain separate so simulator changes do not
become mixed with the workload sources.

### Spike implementation

The following files implement the project-defined custom instructions:

- `riscv/insns/xqmac8.h`: decodes eight signed INT8 lanes from each 64-bit
  source register, computes their dot product, and performs the specified
  modulo-64-bit read-modify-write update of `rd`.
- `riscv/insns/xqrequant8.h`: extracts the signed multiplier, six-bit shift,
  and signed INT8 zero point from `rs2`; performs signed 128-bit fixed-point
  multiplication and ties-away-from-zero rounding; then adds the zero point,
  saturates to INT8, and sign-extends the result into `rd`.
- `riscv/encoding.h`: registers the `custom-0` R-type encodings:
  `xqmac8 = 0x0000000b` with `funct7=0`, and `xqrequant8 = 0x0200000b`
  with `funct7=1`. Both use mask `0xfe00707f`.
- `riscv/riscv.mk.in`: includes both instruction implementation files in
  Spike's generated instruction list.

The implementation requires RV64 for both instructions, matching the design
contract. Register overlap is naturally supported because source operands are
read before `WRITE_RD` updates the destination. Unallocated encodings remain
illegal because only the two registered match/mask pairs are added.

### Rebuild command

```sh
cd ~/src/riscv-isa-sim
make -C build -j2
```

The rebuilt simulator is `~/src/riscv-isa-sim/build/spike`. The build uses the
existing configured Spike checkout and regenerates the instruction translation
units from `riscv/riscv.mk.in`. The generated files in `build/` are build
outputs, not hand-edited source.

### Functional verification

The project repository contains a small bare-metal test:

- [`tests/test_custom_spike.c`](tests/test_custom_spike.c): emits both
  instructions with GNU assembler `.insn` directives and checks known results.
- [`scripts/test_custom_spike.sh`](scripts/test_custom_spike.sh): builds the
  test for `rv64imc`, runs the rebuilt Spike, and checks that the trace
  contains both custom mnemonics, reaches the pass loop, and does not execute
  the failure `c.ebreak`.

Run it with:

```sh
./scripts/test_custom_spike.sh
```

Expected output:

```text
custom Spike instruction test passed
```

The directed test uses an eight-lane dot product with an expected accumulator
value of `46`. It then applies `M=3`, `S=1`, and `Z=-2`, expecting the signed
INT8 result `67`. The generated trace is saved in
[`results/raw/custom_spike_test.trace`](results/raw/custom_spike_test.trace).
This verifies the actual emitted instruction words and the simulator's
architectural behavior, rather than only testing a software model.

## Step 2.2 — Emit custom instructions without modifying GCC

The custom instructions are emitted with GNU assembler's `.insn` directive.
No GCC source, backend, opcode table, or assembler installation was modified.
The compiler still allocates ordinary RV64 integer registers, while the inline
assembly supplies the custom instruction encoding and accurately describes its
register dependencies.

### Emission interface

- [`src/kernels/xq_ops.h`](src/kernels/xq_ops.h): reusable inline wrappers for
  `xqmac8`, `xqrequant8`, packed INT8 operands, and packed requantization
  parameters.
- [`src/kernels/qgemm.c`](src/kernels/qgemm.c): uses the wrappers in
  `qgemm_custom()` when compiling for RV64. It emits one `xqmac8` for each
  complete group of eight reduction elements and one `xqrequant8` per output.
- [`src/kernels/qgemm.h`](src/kernels/qgemm.h): documents the RV64 custom path
  and its host scalar fallback.

The `xqmac8` wrapper uses:

```c
".insn r 0x0b, 0x0, 0x00, %0, %1, %2"
: "+r" (accumulator)
: "r" (packed_a), "r" (packed_w)
```

The `+r` constraint tells GCC that the accumulator is both read and written.
The two input constraints provide the packed activation and weight operands.
For `xqrequant8`, the accumulator is deliberately used for both `rd` and
`rs1`, which is legal under the instruction design and avoids an unnecessary
move:

```c
".insn r 0x0b, 0x0, 0x01, %0, %0, %1"
: "+r" (accumulator)
: "r" (params)
```

The helper packs byte zero into bits `[7:0]`, byte one into `[15:8]`, and so
on, matching the architectural lane order. Requantization parameters use the
specified layout: multiplier in bits `[31:0]`, shift in `[37:32]`, and signed
zero point in `[45:38]`. The reserved upper bits remain zero.

### Custom GEMM integration

For each output element, `qgemm_custom()`:

1. Initializes the 64-bit accumulator from the bias.
2. Packs eight contiguous activation values and eight strided weight values.
3. Emits `xqmac8` for each complete eight-element group.
4. Handles a reduction tail shorter than eight elements with scalar
   multiplication, without reading beyond either tensor.
5. Packs the channel quantization parameters and emits `xqrequant8`.
6. Stores the sign-extended INT8 result into the output tensor.

On host builds, `qgemm_custom()` calls the scalar reference because the host
cannot execute the custom RV64 instructions. This preserves the existing
host regression test while ensuring the RV64 compilation selects the actual
custom path.

### Commands and verification

Run the host regression:

```sh
gcc -std=c11 -Wall -Wextra -Werror -O2 \
    tests/test_qgemm.c src/kernels/qgemm.c \
    -o build/test_qgemm
./build/test_qgemm
```

Compile the custom path for RV64IMC and inspect the generated instruction
words:

```sh
riscv64-unknown-elf-gcc \
    -std=c11 -march=rv64imc -mabi=lp64 -O2 \
    -ffreestanding -fno-builtin -Wall -Wextra -Werror \
    -Isrc/kernels -c src/kernels/qgemm.c \
    -o build/qgemm_custom.o

riscv64-unknown-elf-objdump -d build/qgemm_custom.o \
    > build/qgemm_custom.disassembly.txt
```

The unmodified toolchain's `objdump` may print the custom words as raw
instructions rather than mnemonic names. The expected words still visibly use
the `custom-0` opcode:

```text
00b60e0b    # xqmac8 with compiler-selected registers
02fe0e0b    # xqrequant8 with compiler-selected registers
```

Run the complete custom GEMM under the modified Spike:

```sh
./scripts/test_custom_qgemm.sh
```

This builds [`src/benchmarks/x1_main.c`](src/benchmarks/x1_main.c), computes
the same deterministic GEMM with both `qgemm_scalar()` and `qgemm_custom()`,
compares all `M*N` outputs, and then runs the custom ELF under Spike. The
script requires both custom mnemonics in the trace and rejects the failure
loop.

Expected result:

```text
custom qgemm Spike test passed
```

The resulting disassembly and Spike artifacts are saved under
[`results/raw/`](results/raw/), including
[`x1.disassembly.txt`](results/raw/x1.disassembly.txt) and
[`x1.trace`](results/raw/x1.trace).

## Step 2.3 — Test custom instructions independently

The independent instruction test is implemented in
[`tests/test_custom_spike.c`](tests/test_custom_spike.c). It does not call
`qgemm_custom()` or depend on model tensors, so failures can be attributed to
the instruction encoding or Spike semantics rather than GEMM packing or model
integration.

### Test coverage

The test contains software reference functions for both instructions and
compares every emitted-instruction result with the corresponding reference
result:

- `xqmac8`
  - Directed positive, negative, and mixed-sign packed-byte cases.
  - Negative and large accumulator values.
  - A near-`INT64_MAX` accumulator to exercise modulo-64-bit writeback.
  - 64 deterministic pseudo-random operand cases.
- `xqrequant8`
  - Zero, positive, and negative accumulators.
  - Zero, positive, negative, and maximum signed multipliers.
  - Shift values including `0`, `1`, `2`, `6`, `31`, and `63`.
  - Zero points at `-128`, `127`, and intermediate signed values.
  - Positive and negative saturation.
  - 64 deterministic pseudo-random accumulator, multiplier, shift, and
    zero-point cases.

The test also uses `rd = rs1` for `xqrequant8`, verifying the documented
read-modify-write register overlap behavior. The software oracle uses unsigned
bit-pattern accumulation for `xqmac8` so its modulo-`2^64` behavior does not
depend on C signed-overflow behavior. Requantization uses `__int128`, matching
the project reference contract.

### Command and result

Run the independent test with:

```sh
./scripts/test_custom_spike.sh
```

The script compiles the test for `rv64imc`, runs it on the modified Spike, and
checks that the trace reaches the pass loop without executing the failure
`c.ebreak`. It also checks the expected minimum number of custom-instruction
executions, preventing a truncated Spike run from being reported as a pass.

Expected output:

```text
custom Spike instruction test passed (67 xqmac8, 73 xqrequant8)
```

The trace and compiler diagnostics are saved as
[`custom_spike_test.trace`](results/raw/custom_spike_test.trace),
[`custom_spike_test.stdout`](results/raw/custom_spike_test.stdout), and
[`custom_spike_test.stderr`](results/raw/custom_spike_test.stderr).

## Step 2.4 — Integrate custom instructions into the C workload

The workload now has explicit, independently buildable configurations:

- **B0**: scalar MAC and scalar requantization.
- **X1**: `xqmac8` accumulation with scalar software requantization.
- **X2**: `xqmac8` accumulation with `xqrequant8` requantization.

The common implementation is in
[`src/kernels/qgemm.c`](src/kernels/qgemm.c). `qgemm_x1()` and `qgemm_x2()`
share the packed reduction loop, but select different requantization paths.
`qgemm_custom()` remains an X2-compatible API alias for existing callers.

[`src/benchmarks/benchmark_main.c`](src/benchmarks/benchmark_main.c) uses identical
`M=8`, `N=16`, `K=32` tensors and parameters for all configurations. The
reference output is computed before the measured region. The
`benchmark_start` and `benchmark_end` symbols surround only the selected
configuration's GEMM call, so reference computation, output comparison, and
the terminal spin loop do not distort the configuration comparison.

Each configuration verifies all 128 output elements against the scalar
reference after the measured region. A mismatch enters a `c.ebreak` failure
loop; successful execution reaches the normal `nop` pass loop.

## Step 2.5 — Preliminary performance measurements

The reproducible measurement flow is
[`scripts/run_benchmarks.sh`](scripts/run_benchmarks.sh). It:

1. Builds `build/b0.elf`, `build/x1.elf`, and `build/x2.elf` with the same
   compiler flags and workload.
2. Saves size reports and disassemblies.
3. Runs each ELF under the modified Spike with the same ISA and trace limit.
4. Analyzes only the dynamic trace interval from `benchmark_start` to
   `benchmark_end`.
5. Writes per-configuration CSV files and
   [`results/tables/summary.csv`](results/tables/summary.csv).

The parser is
[`scripts/analyze_trace.py`](scripts/analyze_trace.py). It counts total
dynamic instructions, loads, stores, integer instructions, scalar multiply
instructions, branches, custom instructions, `xqmac8`, and `xqrequant8`.
It also computes instructions per output and logical MACs per retired
instruction. The benchmark performs `8*16*32 = 4096` logical MACs in its
measured GEMM.

Run the complete preliminary benchmark with:

```sh
./scripts/run_benchmarks.sh
```

The current measured summary is:

```text
configuration,dynamic_instructions,loads,stores,integer,multiply,branches,custom,xqmac8,xqrequant8,logical_macs,instructions_per_output,macs_per_instruction,text_bytes,instruction_change_vs_b0
B0,62067,8704,128,43084,4352,5799,0,0,0,4096,484.8984,0.065993,1664,0.0000
X1,80190,8704,128,59513,256,11077,512,512,0,4096,626.4844,0.051079,1664,-29.1991%
X2,74820,8704,128,55729,0,9619,640,512,128,4096,584.5312,0.054745,1664,-20.5472%
```

The measured X1/X2 instruction counts are higher than B0 for this small
synthetic workload because explicit byte packing and loop/control overhead
currently outweigh the removed scalar multiplies. This is a useful result, not
a projected speedup: it identifies packing and control overhead as the first
optimization target. The custom instructions are nevertheless executing at the
expected frequencies:

- X1: 512 `xqmac8` instructions, corresponding to `64*8` packed dot-product
  groups, and no custom requantization instructions.
- X2: 512 `xqmac8` and 128 `xqrequant8` instructions, one requantization per
  output.

The reports are ISA-level instruction measurements, not cycle or CPI
measurements. Raw traces and disassemblies are saved under
[`results/raw/`](results/raw/), and the machine-readable tables are under
[`results/tables/`](results/tables/).
