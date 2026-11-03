# Porting the Quantized ResNet-18 Model

This document defines the complete workflow for porting the static INT8
ResNet-18 model from the LiteRT Community repository to the project's
bare-metal RV64IMC workload.

The target model is:

```text
Repository: litert-community/resnet18
File:       resnet18_int8_channelwise.tflite
Format:     LiteRT/TFLite FlatBuffer
Quantization: static W8A8
Weights:    signed INT8, symmetric channelwise quantization
Activations: signed INT8, asymmetric quantization
```

The model page is:

<https://huggingface.co/litert-community/resnet18>

The repository also contains a floating-point model and a weight-only
quantized model. Do not use either of those for the final integer workload:

| File | Use |
|---|---|
| `resnet18.tflite` | Host floating-point reference only |
| `resnet18_dynamic_wi8_afp32.tflite` | Not suitable: activations remain FP32 |
| `resnet18_int8_channelwise.tflite` | Final W8A8 bare-metal workload |

The synthetic GEMM remains useful as an instruction bring-up test, but it is
not the model-porting deliverable.

## 1. Freeze the model and tool versions

### Purpose

This step fixes the exact model inputs and toolchain used by all later
inspection, export, compilation, simulation, and measurement steps. A model
repository's `main` branch can change, and different compiler or simulator
versions can produce different graph details, instruction sequences, or
measurements. Keeping the downloaded files and hashes in the repository makes
the later results auditable and prevents an accidental model replacement.

The selected model and its repository README are stored together:

```text
model/source/resnet18/
├── README.md
├── model_metadata.json
└── resnet18_int8_channelwise.tflite
```

### Implementation

Run the repository script to download the selected model and the README
describing its provenance:

```sh
./scripts/freeze_resnet18.sh
```

The script creates `model/source/resnet18/`, downloads both source files,
verifies their SHA-256 values against the frozen expected values, and writes
the tool and source metadata. It downloads into a temporary file and moves
the file into place only after verification, so a failed or partial download
does not replace a valid local artifact. The large `.tflite` file is ignored
by Git; it can be recreated by running the script after a clean checkout.

The script requires `curl`, `sha256sum`, `python3`, `git`, the modified Spike
executable, and its source checkout. Override the local Spike paths when
necessary:

```sh
SPIKE_BIN=/path/to/spike \
SPIKE_SOURCE=/path/to/riscv-isa-sim \
./scripts/freeze_resnet18.sh
```

The model and README URLs, expected hashes, and model revision can also be
overridden for a deliberately different frozen artifact:

```sh
MODEL_REVISION=<revision> \
MODEL_URL=<model-url> \
README_URL=<readme-url> \
MODEL_SHA256=<sha256> \
README_SHA256=<sha256> \
./scripts/freeze_resnet18.sh
```

Do not override the hashes casually: changing them changes the model
identity and must be documented and reviewed.

The resulting hashes from this acquisition are:

```text
resnet18_int8_channelwise.tflite
8fea1acb62829e120b4953b6fbf0f842e0032909254f51c2d989703b66534683

README.md
b7f5d7d636ff1f50b6b1b4bf870e143e67d563ab3352666a867a04d950180d68
```

Record the model source, URLs, acquisition timestamp, hashes, and tool
versions in
[`model/source/resnet18/model_metadata.json`](model/source/resnet18/model_metadata.json).
The metadata file is the machine-readable freeze record for this step. It
records the current repository revision as `main`; the SHA-256 values identify
the exact downloaded contents even if the remote branch later advances.

The tool versions were captured without installing additional packages:

```sh
python3 --version
riscv64-unknown-elf-gcc --version
/home/andy/src/riscv-isa-sim/build/spike -h
python3 - <<'PY'
from importlib.metadata import version, PackageNotFoundError
for name in ("numpy", "Pillow", "huggingface_hub",
             "ai-edge-litert", "tensorflow"):
    try:
        print(f"{name}={version(name)}")
    except PackageNotFoundError:
        print(f"{name}=not-installed")
PY
```

The Spike executable does not support `--version`; its `-h` output identifies
it as `Spike RISC-V ISA Simulator 1.1.1-dev`. The metadata also records the
Spike source commit and the required `midterm-custom-isa` branch because the
custom instructions are not present in an unmodified Spike build.

The currently captured host package state includes TensorFlow, NumPy, and
Pillow. `huggingface_hub` and `ai-edge-litert` are recorded as
`not-installed`; they are not needed to download the files with `curl`, but
will be required if the later host-side workflow uses the model-page Python
example or those specific APIs. The later environment-preparation step must
install and record them before model inspection.

The model page notes that the model may carry license terms derived from
PyTorch Vision and ImageNet. Record and follow those terms before
redistributing the model or generated weights.

## 2. Prepare the host Python environment

### Purpose

The host-side tools inspect the LiteRT FlatBuffer, execute the original model
for reference outputs, and generate static C data for the bare-metal
executor. They are not linked into the RV64 target. Keeping them in a
project-local virtual environment prevents host package changes from
affecting the system Python installation or the RISC-V build.

The required packages are:

- `numpy`: tensor and quantization-array handling.
- `Pillow`: image loading and preprocessing for model-level tests.
- `huggingface_hub`: optional authenticated/reproducible access to the model
  repository.
- `ai-edge-litert`: LiteRT model loading and host reference execution.

### Implementation

Create a virtual environment and install the host dependencies:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install numpy Pillow huggingface_hub ai-edge-litert
```

`python3 -m venv .venv` creates an isolated environment without changing the
system interpreter. Calling `.venv/bin/python` explicitly makes each command
independent of shell activation state and ensures that packages are installed
into the intended environment. `pip install --upgrade pip` updates only the
virtual environment's package manager.

Record the complete resolved dependency set:

```sh
.venv/bin/python -m pip freeze --all \
  > model/source/resnet18/host_requirements.txt
```

The resulting
[`host_requirements.txt`](model/source/resnet18/host_requirements.txt) is a
reproducibility record for host model inspection and reference execution. The
`.venv/` directory is ignored by Git because it contains platform-specific
installed binaries; the requirements file is the portable record.

Verify the environment before proceeding:

```sh
.venv/bin/python - <<'PY'
import sys
from importlib.metadata import version

print(f"python={sys.version.split()[0]}")
for package in ("numpy", "Pillow", "huggingface_hub", "ai-edge-litert"):
    print(f"{package}={version(package)}")
PY
```

The verified environment currently contains:

```text
Python 3.10.12
numpy 2.2.6
Pillow 12.3.0
huggingface_hub 2.4.0
ai-edge-litert 2.3.0
```

This step does not install TensorFlow into the virtual environment because
the selected host workflow uses `ai-edge-litert`. TensorFlow remains useful
only if a later exporter or reference tool specifically requires its APIs.
The host environment is intentionally separate from the bare-metal command,
which continues to use `riscv64-unknown-elf-gcc` with `-ffreestanding` and
without a host runtime.

## 3. Inspect the LiteRT graph before implementing operators

### Purpose

This step converts the opaque LiteRT FlatBuffer into an explicit graph
inventory before any bare-metal operator is implemented. It identifies the
tensor layout, signedness, quantization parameters, constant-buffer sizes,
operator order, convolution options, and the exact operator set that the
fixed executor must support. Inspecting first prevents an incorrect
implementation based on assumptions from the synthetic GEMM or from a
different ResNet export.

### Implementation

[`scripts/inspect_litert_model.py`](scripts/inspect_litert_model.py) uses the
LiteRT interpreter for public tensor metadata and the installed generated
FlatBuffer schema for operator options. It:

- Input and output tensor IDs, shapes, types, scales, and zero points.
- Every operator in execution order.
- Tensor shapes and buffer sizes.
- Weight quantization dimensions and per-channel scales.
- Bias types and quantization.
- Convolution stride, padding, dilation, and fused activation.
- Operators that are not supported by the planned bare-metal executor.

The script writes two artifacts:

- [`graph_report.json`](model/source/resnet18/graph_report.json): complete
  machine-readable tensor and operator report.
- [`graph_summary.txt`](model/source/resnet18/graph_summary.txt): concise
  human-readable inventory for review.

Run it with the project virtual environment:

```sh
PYTHONPATH="$PWD" \
  .venv/bin/python scripts/inspect_litert_model.py \
    --model model/source/resnet18/resnet18_int8_channelwise.tflite \
    --json model/source/resnet18/graph_report.json \
    --summary model/source/resnet18/graph_summary.txt
```

`PYTHONPATH="$PWD"` makes the repository root explicit when the script is
launched by path from the `scripts/` directory. The `--model` option selects
the frozen FlatBuffer, while `--json` and `--summary` make the inspection
results persistent rather than leaving them only in terminal output.

The script disables interpreter delegates for inspection. An interpreter
delegate is an optional backend that replaces supported parts of the graph
with an optimized implementation, such as a CPU, GPU, or accelerator kernel.
Delegates are useful for host execution, but their synthetic graph entries do
not represent operators that the bare-metal executor must implement. The
script therefore reads the actual FlatBuffer subgraph length, so only model
operations are reported and host-added `DELEGATE` entries are excluded.

### Inspection results

The selected model contains one subgraph, 86 tensors, and 37 model
operators:

```text
TRANSPOSE          1
PAD                5
CONV_2D           20
MAX_POOL_2D        1
ADD                8
AVERAGE_POOL_2D    1
FULLY_CONNECTED    1
```

No operator is outside the currently planned executor set. The model input
is signed INT8 in NCHW layout. NCHW means
`[batch, channels, height, width]`; here there is one image, three color
channels, and a 224x224 spatial resolution. The graph immediately transposes
this to NHWC, which means `[batch, height, width, channels]`, because the
following LiteRT convolution operators use that layout.

```text
shape:       [1, 3, 224, 224]
scale:       0.01865844801068306
zero point:  -14
```

The shape specifies the number of elements in each tensor dimension. The
quantization scale and zero point map an INT8 value `q` to an approximate real
value using `scale * (q - zero_point)`. Thus, the input zero point of `-14`
means that quantized value `-14` represents real zero; the scale determines
the real-value spacing between adjacent quantized values.

The output is a signed INT8 ImageNet logits tensor:

```text
shape:       [1, 1000]
scale:       0.1705460548400879
zero point:  -61
```

The output shape contains one batch and 1,000 class scores. Its scale and zero
point must be used when interpreting the final logits; the bare-metal program
must preserve the quantized values rather than treating them as unscaled
floating-point scores.

Convolution weights are channelwise quantized along dimension zero. For
example, the first convolution has shape `[64, 7, 7, 3]` and 64 weight
scales. In LiteRT's convolution layout, this means
`[output_channels, kernel_height, kernel_width, input_channels]`: 64 filters,
each with a 7x7 kernel over three input channels. A later 64-channel
convolution has shape `[64, 3, 3, 64]`: 64 output filters, each with a 3x3
kernel over 64 input channels, and therefore also has 64 per-output-channel
scales. The report shows the complete scale arrays and quantized dimensions
needed by the exporter.

The graph includes an initial NCHW-to-NHWC `TRANSPOSE`, explicit `PAD`
operators, fused ReLU on the first convolution and residual-path
convolutions, max pooling, eight quantized residual `ADD` operations, global
average pooling, and the final fully connected layer. The executor must
preserve this order and cannot assume that padding or activation is always
fused into convolution.

Do not assume that the model uses the same layout or quantization convention as
the synthetic GEMM. The integer conversion rule is:

```text
real_value = scale * (quantized_value - zero_point)
```

The expected ResNet graph contains convolution, pooling, activation, residual
addition, reshape, average pooling, and fully connected operations. Confirm
the actual operator list from the downloaded model before writing the executor.

The final bare-metal implementation should support only the operators present
in this fixed graph; a general-purpose LiteRT runtime is not required.

## 4. Validate the original model on the host

Before exporting tensors, run the original model with the host LiteRT
interpreter. Use deterministic inputs first, then add representative
ImageNet-style images.

For each test input, save:

- The input tensor after preprocessing and quantization.
- The final output tensor.
- The predicted class, if using an image.
- Intermediate tensors when the host API makes them available.

The model page's example uses RGB preprocessing with a 224x224 center crop.
Use the actual input tensor shape, type, scale, and zero point reported by the
inspection script rather than copying preprocessing assumptions blindly.

The host reference is the authority for validating the exported C graph.

## 5. Export static C model data

Create `scripts/export_litert_model.py`. It should read the FlatBuffer and
generate:

```text
model/exported/
├── resnet18_graph.h
├── resnet18_weights.h
├── resnet18_biases.h
├── resnet18_shapes.h
├── resnet18_quant_params.h
└── model_metadata.json
```

Also generate deterministic test vectors:

```text
model/test_vectors/
├── input_0.bin
├── reference_output_0.bin
└── intermediate_*.bin
```

The exported graph metadata must include, for every operator:

- Input and output tensor IDs.
- Weight and bias tensor IDs.
- Tensor dimensions and layout.
- Stride, padding, dilation, and kernel size.
- Fused activation.
- Input and output scales and zero points.
- Per-channel weight scales and zero points.
- Fixed-point multiplier and shift used by the target executor.

Store weights as aligned signed `int8_t` arrays and biases in the type used by
the model, normally `int32_t`. Use read-only model-data sections where
possible.

Example generated declarations:

```c
extern const int8_t resnet18_conv0_weights[];
extern const int32_t resnet18_conv0_bias[];
extern const int32_t resnet18_conv0_multiplier[];
extern const uint8_t resnet18_conv0_shift[];
extern const int32_t resnet18_conv0_weight_sum[];
```

The weight-sum type must be wide enough for the largest channel reduction; it
must not be restricted to `int8_t` if the sum can exceed that range.

## 6. Convert LiteRT quantization parameters

For a quantized convolution with symmetric weights:

```text
acc =
    bias
    + Σ ((input_q - input_zero_point) * weight_q)
```

The current `xqmac8` instruction computes:

```text
Σ (input_q * weight_q)
```

Therefore the kernel must apply the activation zero-point correction:

```text
acc =
    bias
    + Σ (input_q * weight_q)
    - input_zero_point * Σ weight_q
```

Precompute the signed weight sum for every output channel during export.
If the model reports a nonzero weight zero point, use the general expansion:

```text
Σ ((input_q - input_zp) * (weight_q - weight_zp))
```

and add the required correction terms. Do not silently assume symmetric
weights when the model metadata says otherwise.

For each output channel, calculate:

```text
real_multiplier =
    input_scale * weight_scale[channel] / output_scale
```

Represent this as:

```text
real_multiplier ≈ M / 2^S
```

where `M` and `S` are the multiplier and shift fields consumed by
`xqrequant8`. Preserve the model's output zero point and signed INT8
saturation range.

The exporter must reject a multiplier or shift that cannot be represented by
the instruction format. It must not silently clamp parameters. Use a scalar
fallback or revise the instruction format if an actual model layer requires a
larger range.

Compare the rounding behavior with the LiteRT reference. The current ISA
contract uses round-to-nearest, ties-away-from-zero, followed by zero-point
addition and saturation to `[-128, 127]`.

## 7. Implement the scalar bare-metal executor first

Implement B0 before emitting custom instructions. The executor should contain
only the operators used by the inspected graph:

```c
void conv2d_scalar(...);
void fully_connected_scalar(...);
void max_pool_scalar(...);
void average_pool_scalar(...);
void add_scalar(...);
void relu_scalar(...);
```

Required constraints:

- Static or global model and activation storage.
- No `malloc`, file I/O, operating system, or `pk`.
- No floating-point instructions.
- Reusable activation buffers.
- Explicit tensor dimensions and strides.
- Exact integer quantization behavior.

Use two or more *ping-pong activation buffers* where possible. Do not place the
multi-megabyte model or full activation tensors on the stack.

The complete ResNet-18 graph is approximately:

```text
input
→ initial convolution
→ ReLU
→ max pool
→ residual stages 1–4
→ global average pool
→ fully connected layer
→ output
```

Batch normalization should normally already be folded into the exported
convolution weights and biases. Verify this from the graph and do not add a
runtime floating-point batch-normalization operator.

## 8. Choose the convolution implementation

Initially, implement convolution with `im2col` plus the existing GEMM path
because it simplifies correctness reuse:

```text
im2col(input) × weight_matrix = output_matrix
```

The reduction length is:

```text
kernel_height * kernel_width * input_channels
```

For later ResNet layers this is usually divisible by eight. The first
convolution and any other non-divisible reduction must use a scalar tail.

After correctness is established, consider a direct convolution kernel if the
`im2col` buffer creates excessive memory traffic. The benchmark must include
the chosen transformation and data movement if they are part of the workload.

## 9. Implement residual addition and activation

Residual addition is quantized arithmetic, not necessarily an `int8_t` wrap:

```text
main_real = main_scale * (main_q - main_zp)
skip_real = skip_scale * (skip_q - skip_zp)
sum_real = main_real + skip_real
output_q = round(sum_real / output_scale) + output_zp
```

If the graph has equal input and output quantization parameters, use the
corresponding simplified integer expression only after verifying the metadata.
Apply the graph's fused ReLU after the correctly quantized addition.

## 10. Add X1 and X2 kernels

After B0 matches the host model, add the custom paths:

### X1

Use `xqmac8` for groups of eight signed INT8 products and retain scalar
requantization:

```text
packed input and weight loads
→ xqmac8 accumulation
→ activation zero-point correction
→ scalar requantization
```

### X2

Use both custom instructions:

```text
packed input and weight loads
→ xqmac8 accumulation
→ activation zero-point correction
→ xqrequant8
```

Load channel-specific multiplier, shift, and output zero-point parameters for
each output channel. Use a scalar tail whenever the reduction length is not
divisible by eight.

The custom paths must produce exactly the same output tensors as B0 for the
same input and model parameters.

## 11. Validate layer by layer

Do not debug only from the final ImageNet class. For each layer:

1. Run the original LiteRT model on the host.
2. Run the exported scalar operator on the host.
3. Compare the complete output tensor.
4. Run the corresponding RV64 operator under Spike.
5. Compare the RV64 output with the host reference.
6. Repeat for X1 and X2.

Report:

```text
element mismatch count
maximum absolute error
first mismatch tensor and index
expected value
actual value
```

Common causes of mismatches are signedness, tensor layout, padding,
activation zero-point correction, per-channel scales, bias scaling, rounding,
residual-add quantization, and incorrect placement of ReLU.

## 12. Build the bare-metal model

Build separate configurations so the measured code differs only in the
selected kernel implementation:

```text
build/resnet18_b0.elf
build/resnet18_x1.elf
build/resnet18_x2.elf
```

Use the existing startup and linker structure:

```sh
riscv64-unknown-elf-gcc \
    -std=c11 \
    -march=rv64imc \
    -mabi=lp64 \
    -mcmodel=medany \
    -O2 \
    -ffreestanding \
    -fno-builtin \
    -nostdlib \
    -nostartfiles \
    -Wall -Wextra -Werror \
    -Isrc/kernels -Imodel/exported \
    src/benchmarks/startup.S \
    src/benchmarks/resnet18_main.c \
    src/kernels/resnet18_ops.c \
    src/kernels/qgemm.c \
    -T src/benchmarks/linker.ld \
    -o build/resnet18_b0.elf
```

For custom builds, enable X1 or X2 through the same compile-time mechanism
used by the existing benchmark. Keep `-march=rv64imc`; do not add `F`, `D`,
or vector extensions.

Inspect the disassembly and confirm that B0 contains no floating-point
instructions and that X1/X2 contain the expected custom instruction words:

```sh
riscv64-unknown-elf-objdump -d build/resnet18_x2.elf
```

## 13. Run the model under modified Spike

Use the Spike executable built from the `midterm-custom-isa` branch:

```sh
/home/andy/src/riscv-isa-sim/build/spike \
    --isa=RV64IMC \
    -l \
    build/resnet18_b0.elf \
    > results/raw/resnet18_b0.trace \
    2> results/raw/resnet18_b0.stderr
```

Run the same input and model under B0, X1, and X2. The bare-metal program
must write a deterministic pass/fail result and enter the normal pass loop.
Failures must enter the existing `c.ebreak` failure loop.

Use `benchmark_start` and `benchmark_end` symbols around the model inference
call. Exclude startup, reference generation, output comparison, and terminal
loops from kernel-only measurements. Also report full-model counts when
feasible; do not compare mismatched measurement regions.

## 14. Measure the complete model

For every configuration, record:

```text
total dynamic instructions
loads and stores
integer ALU instructions
scalar multiply instructions
branches and jumps
xqmac8 count
xqrequant8 count
logical MAC count
instructions per output
MACs per retired instruction
custom-instruction frequency
static text size
weight and activation memory
```

At minimum, report separate rows for:

```text
initial convolution
one normal residual stage
one downsampling residual stage
final fully connected layer
complete ResNet-18 inference
```

Retain raw traces, disassemblies, program output, compiler diagnostics, and
machine-readable CSV summaries. Spike traces provide ISA-level measurements;
they do not prove cycle speedup, CPI improvement, or hardware timing.

## 15. Automate the workflow

The final repository should provide commands equivalent to:

```sh
./scripts/fetch_resnet18_model.sh
python3 scripts/inspect_litert_model.py \
    --model model/source/resnet18/resnet18_int8_channelwise.tflite
python3 scripts/export_litert_model.py \
    --model model/source/resnet18/resnet18_int8_channelwise.tflite \
    --out-dir model/exported \
    --test-dir model/test_vectors
./scripts/run_resnet18_benchmarks.sh
```

The benchmark script must:

1. Verify the model file and recorded hash.
2. Build B0, X1, and X2.
3. Run host and instruction-level correctness tests.
4. Run all configurations under modified Spike.
5. Compare final and, where available, intermediate outputs.
6. Save traces and disassemblies.
7. Invoke the trace analyzer.
8. Generate a consolidated CSV.
9. Return a nonzero status on any build, simulation, or verification failure.

Do not document commands until they work from a clean checkout with the
required external model and toolchain explicitly identified.

## 16. Recommended implementation order

1. Download and hash `resnet18_int8_channelwise.tflite`.
2. Inspect its graph and quantization metadata.
3. Validate the original host model.
4. Export one convolution and its test vectors.
5. Implement and validate one scalar convolution.
6. Add X1 and X2 for that convolution.
7. Implement residual addition, pooling, and activation.
8. Validate one complete residual block.
9. Implement the complete scalar ResNet-18 graph.
10. Validate full B0 inference layer by layer.
11. Enable X1 and X2 for the major convolution and fully connected kernels.
12. Validate full X1/X2 inference against B0 and the host reference.
13. Measure per-layer and complete-model instruction counts.
14. Update the repository README and prepare the presentation.

## 17. Acceptance criteria

The model-porting work is complete when:

- The exact model file and hash are recorded.
- The graph and quantization metadata are exported reproducibly.
- B0 runs the complete selected graph without an operating system or `pk`.
- X1 and X2 execute the custom instructions in the major dot-product kernels.
- Final outputs match the host reference within the documented integer rule.
- No floating-point extension is required.
- Per-layer and complete-model benchmark results are saved.
- B0/X1/X2 measurements use identical inputs and measurement boundaries.
- README commands reproduce the export, build, verification, and benchmark flow.
