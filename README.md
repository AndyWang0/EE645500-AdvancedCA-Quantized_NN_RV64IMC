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
