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

### Purpose

This step establishes the numerical reference for every later exporter,
scalar-kernel, and custom-instruction comparison. The host LiteRT execution
is not the bare-metal implementation; it is the trusted reference used to
detect graph, layout, signedness, padding, and quantization mistakes before
porting the model.

Use deterministic inputs first so that a failure can be reproduced exactly.
After the integer port is correct, add representative ImageNet-style images
to validate preprocessing and classification behavior.

### Implementation

[`scripts/validate_litert_model.py`](scripts/validate_litert_model.py)
creates one deterministic signed-INT8 input using NumPy's seeded generator,
runs the frozen model with the host LiteRT interpreter, and saves:

- `input_0.bin`: the exact quantized input bytes.
- `reference_output_0.bin`: the final quantized output bytes.
- `validation_metadata.json`: tensor names, shapes, dtypes, quantization
  parameters, statistics, and predicted class index.
- `intermediate/tensor_<id>.bin`: available tensor snapshots after
  invocation.

Run the deterministic validation with:

```sh
PYTHONPATH="$PWD" \
  .venv/bin/python scripts/validate_litert_model.py \
    --model model/source/resnet18/resnet18_int8_channelwise.tflite \
    --out-dir model/test_vectors/resnet18 \
    --debug-dir model/test_vectors/host_validation/intermediate
```

The default seed is `20261104`; pass `--seed` to create a different
reproducible test vector. The script checks that the model has exactly one
INT8 input and one output, uses the model's actual input shape, invokes the
interpreter, and writes raw arrays with NumPy's native contiguous layout.
It enables LiteRT's `experimental_preserve_all_tensors` debugging mode, which
keeps intermediate buffers readable after invocation. It still records any
tensor that remains unavailable explicitly instead of silently omitting it.

The current run produced:

```text
Host validation passed
Predicted class index: 107
Recorded 86 tensor records (86 snapshots available)
```

To compare a ported run with these host snapshots, place its tensor dumps and
matching `validation_metadata.json` in a separate directory using the same
`tensor_<id>.bin` naming convention, then run:

```sh
.venv/bin/python scripts/compare_resnet18.py \
  --host-dir model/test_vectors/resnet18 \
  --ported-dir build/resnet18_b0_tensors
```

The comparator checks every available tensor, reports shape errors, mismatch
counts, maximum absolute error, and the first mismatching element, and exits
with status 1 if any tensor differs. This makes it suitable for B0, X1, and
X2 regression scripts. The target-side dump producer can use
`resnet18_copy_tensor()` to serialize each nonconstant tensor into this
directory format; the comparison logic is intentionally independent of
whether the tensors came from a host executor, Spike, or a hardware run.

The intermediate directory is ignored by Git because it contains generated
per-tensor binaries. The input, final reference output, and validation
metadata remain the compact reproducibility artifacts.

The model page's example uses RGB preprocessing with a 224x224 center crop.
Use the actual input tensor shape, type, scale, and zero point reported by the
inspection script rather than copying preprocessing assumptions blindly.

For image-based validation, preprocess an RGB image according to the model's
documented 224x224 input pipeline, quantize using the inspected input scale
and zero point, and save the resulting input bytes through the same script
interface. Do not feed normalized floating-point data to the INT8 interpreter.

The host reference is the authority for validating the exported C graph.

## 5. Export static C model data

### Purpose

The LiteRT FlatBuffer is not a bare-metal executable and should not be parsed
at runtime for this fixed workload. This step converts its graph metadata,
constant tensors, and quantization parameters into ordinary C headers that
can be compiled with `riscv64-unknown-elf-gcc`. The generated files make the
model self-contained while preserving the original tensor IDs and operator
order for the later static executor.

### Implementation

[`scripts/export_litert_model.py`](scripts/export_litert_model.py) reads the
FlatBuffer with the generated LiteRT schema, extracts each non-empty constant
buffer, preserves its declared scalar type and shape, and emits:

```text
model/exported/
└── resnet18/
    ├── resnet18_graph.h
    ├── resnet18_weights.h
    ├── resnet18_biases.h
    ├── resnet18_shapes.h
    ├── resnet18_quant_params.h
    └── model_metadata.json
```

Run it with:

```sh
PYTHONPATH="$PWD" \
  .venv/bin/python scripts/export_litert_model.py \
    --model model/source/resnet18/resnet18_int8_channelwise.tflite \
    --out-dir model/exported/resnet18 \
    --test-dir model/test_vectors/resnet18
```

The exporter produces aligned read-only arrays. INT8 constants are placed in
`resnet18_weights.h`, INT32 constants in `resnet18_biases.h`, tensor shapes in
`resnet18_shapes.h`, and tensor scales/zero points plus representable fixed
point parameters in `resnet18_quant_params.h`. `resnet18_graph.h` preserves
operator input/output IDs and the graph operation count. The JSON metadata
keeps the complete tensor and operator records for auditing.

The `--test-dir` option copies the compact host reference vectors from the
host-validation directory into the model-specific test-vector directory when
they already exist:

```text
model/test_vectors/
├── resnet18/
│   ├── input_0.bin
│   ├── reference_output_0.bin
│   └── validation_metadata.json
└── host_validation/
    └── intermediate/
```

The canonical input/reference pair is kept beside the model name because the
later bare-metal runner must load exactly those bytes without invoking Python
or LiteRT. The intermediate directory is only a host-side debugging artifact
and is kept separate from the compact fixtures.

The exported graph metadata includes, for every operator:

- Input and output tensor IDs.
- Weight and bias tensor IDs.
- Tensor dimensions and layout.
- Stride, padding, dilation, and kernel size.
- Fused activation.
- Input and output scales and zero points.
- Per-channel weight scales and zero points.
- Fixed-point multiplier and shift used by the target executor.

The current export generated 86 tensor records and 37 operator records. The
large generated ResNet headers are intentionally not Git artifacts; they can
be recreated from the frozen model by rerunning the command. The synthetic
headers remain in `model/exported/` for the existing GEMM bring-up tests.

Fixed-point conversion is recorded but not silently forced. Twenty of the 21
convolution/fully-connected operators have channel multipliers representable
by the current `xqrequant8` fields. The first convolution has eight extremely
small channel multipliers outside the configured shift range. Their channel
indices are recorded in `unrepresentable_channels`, and the exporter also
emits exact wider-range `fallback_multipliers`, `fallback_shifts`, and
`use_scalar_fallback` arrays for Section 6.
This prevents the exporter from silently changing that model channel's
quantization.

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

### Purpose

This step converts the floating-point quantization metadata stored in the
LiteRT FlatBuffer into integer parameters that the bare-metal executor can use
without a runtime quantization library. It also preserves the correction terms
needed when activation and weight zero points are not both zero. The exporter
is the single source of these parameters so that B0, X1, and X2 use the same
model scales and rounding contract.

### 6.1 Read and validate quantization metadata

`export_litert_model.py` reads the input, weight, and output quantization
records for every `CONV_2D` and `FULLY_CONNECTED` operator. It restores each
constant tensor to its declared FlatBuffer shape before processing it. This
matters for channelwise weights: the first convolution is `[64, 7, 7, 3]`,
so dimension zero contains 64 output channels even though the underlying byte
buffer is flat.

For each operator, the exporter accepts either one quantization value
broadcast to all output channels or exactly one value per output channel. It
rejects any other scale or zero-point length instead of silently broadcasting
or truncating metadata. Weight zero points are converted to integer values and
are exported per channel, including the current model's all-zero values.

### 6.2 Export zero-point correction terms

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

The exporter computes the signed sum of the stored INT8 weights independently
for every output channel. For the `[output_channel, kernel_height,
kernel_width, input_channel]` convolution layout, this is a reduction over
dimensions 1 through 3. Fully connected weights use the same dimension-zero
output-channel convention. The generated quantization header contains:

```c
resnet18_op_N_input_zero_point
resnet18_op_N_weight_zero_points[channel]
resnet18_op_N_weight_sums[channel]
```

The sums are emitted as `int32_t`; the largest observed reduction in this
model is within that range, but the accumulation used to calculate them is
`int64_t` so the export cannot overflow while producing the metadata. The
future operator implementation must use these arrays when applying the
activation and weight zero-point correction.

### 6.3 Convert real multipliers to fixed point

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

The current instruction supports shifts from 0 through 63. For a channel
whose exact multiplier needs a larger shift, the exporter emits a wider
fallback pair and a `use_scalar_fallback` flag. The kernel dispatches those
channels to an integer-only scalar path:

```c
if (use_scalar_fallback[channel]) {
    output = requantize_wide(
        accumulator,
        fallback_multipliers[channel],
        fallback_shifts[channel],
        output_zero_point);
} else {
    output = xqrequant8(
        accumulator,
        pack_requant_params(
            multipliers[channel],
            shifts[channel],
            output_zero_point));
}
```

`requantize_wide` must use a signed wide intermediate, round to nearest with
ties away from zero, add the output zero point, and saturate to `[-128, 127]`.
It must preserve the exact fallback multiplier and shift rather than clamp
them to the custom-instruction range. This keeps the model numerically
correct while limiting the scalar fallback to the eight exceptional channels
in the first convolution.

The shared C reference implementation exposes this fallback behavior as
`requantize_scalar_wide()`. The existing
`requantize_reference()` calls the same implementation for compatibility, so
the scalar model path and exceptional-channel fallback cannot silently acquire
different rounding or saturation rules.

The conversion uses `math.frexp()` to normalize each positive real multiplier,
rounds the normalized significand to a signed 31-bit multiplier, and stores
the corresponding power-of-two shift. The exporter computes the ordinary
`xqrequant8` representation with a maximum shift of 63. It separately computes
an exact wider fallback representation with a maximum shift of 126, leaving
one bit of headroom for the signed `__int128` rounding intermediate. A
representable channel receives both forms; an unrepresentable channel keeps
the ordinary fields at zero and sets `use_scalar_fallback[channel]`.

An alternative implementation option is to expand `xqrequant8` so that it can
represent the full fixed-point range required by the model. For example, a
future revision could provide a wider shift field, a wider multiplier format,
or a second requantization encoding with the same rounding, zero-point, and
saturation semantics. This would allow the exceptional channels to remain on
the custom-instruction path and may reduce scalar dispatch overhead. It would
also require a new encoding decision and corresponding updates to the Spike
implementation, inline-assembly wrapper, generated parameter format, tests,
and benchmark results. The current design therefore keeps the existing
`xqrequant8` encoding as the baseline and uses the exact scalar fallback; an
expanded-instruction variant can be evaluated later without changing the
model-export correctness contract.

The exporter must never silently clamp parameters. A scalar fallback is
preferred here because only a small number of channels are affected; changing
the ISA format would require revising the encoding, Spike implementation,
assembly wrapper, tests, and existing measurements.

Compare the rounding behavior with the LiteRT reference. The current ISA
contract uses round-to-nearest, ties-away-from-zero, followed by zero-point
addition and saturation to `[-128, 127]`.

### 6.4 Run and verify the conversion

Regenerate the model-specific export with:

```sh
PYTHONPATH="$PWD" .venv/bin/python scripts/export_litert_model.py \
  --model model/source/resnet18/resnet18_int8_channelwise.tflite \
  --out-dir model/exported/resnet18 \
  --test-dir model/test_vectors/resnet18
```

The export currently produces 21 quantized convolution/fully-connected
operators. Eight channels of operator 2 (the first convolution) require the
wide fallback, with indices `[2, 4, 7, 9, 13, 36, 38, 48]`; all other
channels fit the current instruction fields. The model's weight zero points
are all zero, but they are still emitted so the executor can follow the
general correction formula rather than relying on that observation.

Compile the generated quantization header for the target ABI:

```sh
printf '#include "resnet18_quant_params.h"\nint main(void) { return 0; }\n' \
  > /tmp/check_resnet_quant.c
riscv64-unknown-elf-gcc -std=c11 -ffreestanding \
  -march=rv64imc -mabi=lp64 \
  -I model/exported/resnet18 \
  -c /tmp/check_resnet_quant.c -o /tmp/check_resnet_quant.o
```

This verifies that the generated multiplier, shift, zero-point, and weight-sum
arrays are valid C for the target compiler. Numerical layer comparison is
performed after the scalar executor is implemented in Section 7; this step
verifies conversion and metadata preservation.

## 7. Implement the scalar bare-metal executor first

The scalar executor is the B0 correctness baseline. It runs the complete
inspected graph without custom instructions, dynamic allocation, an operating
system, LiteRT, or floating-point arithmetic. X1 and X2 are only meaningful
after this path produces the same tensors as the host reference.

### 7.1 Requirements and static storage

The executor must satisfy these constraints:

- Static or global model and activation storage.
- No `malloc`, file I/O, operating system, or `pk`.
- Integer-only executor arithmetic; the target is compiled for `rv64imc`
  without `F` or `D`.
- Activation storage that can eventually be reused after a tensor's last
  consumer.
- Explicit tensor dimensions and strides.
- Exact integer quantization behavior.

The exporter generates `resnet18_graph.h` and `resnet18_graph.c` in
`model/exported/resnet18/`. The header contains declarations for every
nonconstant tensor, pointers to generated INT8 weights and INT32 biases,
tensor ranks and dimensions, element counts, zero points, and operator
metadata; the C file owns the single shared tensor-storage and graph-table
definitions. This prevents separate translation units from accidentally
comparing or updating different copies, and keeps model data in `.rodata` and
activations in `.bss`.
Complete tensors are not placed on the stack. The current implementation
allocates one static storage array for each nonconstant tensor; this is
correctness-first storage, not yet a true ping-pong allocation. A later
memory-optimization pass can reuse buffers after the last consumer of each
tensor, but it must preserve the residual branches whose values remain live.

`resnet18_b0_main.c` loads the canonical input fixture through an executor API
so the generated C file's storage has one owner even though the main and
executor are separate translation units. Startup loads `__stack_top` from the
linker script; it must not use a fixed address inside the activation `.bss`.

### 7.2 Implement the operator kernels

`resnet18_b0_executor.c` dispatches the 37 operators in FlatBuffer order:

```text
TRANSPOSE
PAD
CONV_2D
MAX_POOL_2D
ADD
AVERAGE_POOL_2D
FULLY_CONNECTED
```

The transpose converts the model's NCHW input `[1, 3, 224, 224]` to the NHWC
layout used by the convolution graph. Padding uses the tensor's quantized zero
point, not numeric zero. Convolution weights are indexed as
`[output_channel, kernel_height, kernel_width, input_channel]`; SAME padding
is derived from the input/output shapes and VALID padding is handled directly.
The first convolution and fully connected layer use scalar reduction loops,
including the non-divisible reduction tail naturally.

Max pooling performs integer comparisons. Global average pooling accumulates
centered INT8 values in `int64_t`, applies round-to-nearest ties-away-from-zero
division by the filter area, and restores the output zero point. Residual ADD
converts both inputs with their exported fixed-point multipliers, combines the
scaled products before one rounding step, adds the output zero point, saturates
to INT8, and applies fused ReLU in the output quantized domain.

For this model, the global-average-pool input and output use the same scale
and zero point, so the centered integer average is valid. A different model
would require an explicit average-pool output requantization step.

Convolution and fully connected operators use the exported per-channel weight
sums and zero points:

```text
acc =
    bias
    + Σ(input_q * weight_q)
    - input_zp * weight_sum
    - weight_zp * input_sum
    + K * input_zp * weight_zp
```

The result is sent through `requantize_reference()`, or through
`requantize_scalar_wide()` for the eight channels marked by
`use_scalar_fallback`. Fused ReLU clamps to the output tensor's zero point.

The intended scalar-kernel decomposition is:

```c
void conv2d_scalar(...);
void fully_connected_scalar(...);
void max_pool_scalar(...);
void average_pool_scalar(...);
void add_scalar(...);
void relu_scalar(...);
```

In the current implementation these operations are dispatched through static
helpers in `resnet18_b0_executor.c`; some helpers are not yet exposed with the
names above, and fused ReLU is applied inside the convolution and ADD paths
instead of through a separate `relu_scalar()` function. The prototype list
defines the desired decomposition, not a claim that each function is already
a public API.

`qgemm.h` is included because it declares the shared fixed-point
requantization functions used by the executor:
`requantize_reference()` and `requantize_scalar_wide()`. The B0 convolution
and fully connected loops do not currently call `qgemm_scalar()`; the include
is not for GEMM itself. If the executor is later changed to an im2col+GEMM
implementation, the same header will also provide the GEMM declarations.

### 7.3 Reproduce and validate the B0 implementation

Run the host reference first. This regenerates the canonical input, final
output, preserved intermediate tensors, and validation metadata:

```sh
PYTHONPATH="$PWD" .venv/bin/python scripts/validate_litert_model.py \
  --model model/source/resnet18/resnet18_int8_channelwise.tflite \
  --out-dir model/test_vectors/resnet18 \
  --debug-dir model/test_vectors/host_validation/intermediate
```

Build and run the native scalar B0 executor. This creates
`build/resnet18_b0_tensors/tensor_<id>.bin` and a matching metadata file:

```sh
./scripts/build_resnet18_b0_host_dump.sh
```

Compare all tensors produced by B0 against the LiteRT reference:

```sh
.venv/bin/python scripts/compare_resnet18.py \
  --host-dir model/test_vectors/resnet18 \
  --ported-dir build/resnet18_b0_tensors
```

The expected result is:

```text
Compared 38 tensors; mismatching tensors: 0
```

This native comparison must pass before the target build is attempted. It
isolates graph, layout, and quantization errors from RISC-V build or Spike
problems.

### 7.4 Build the B0 executable for RV64IMC

Use the reproducible build script:

```sh
./scripts/build_resnet18_b0.sh
```

The script converts `model/test_vectors/resnet18/input_0.bin` into a temporary
C header, compiles the startup, B0 main, B0 executor, and shared
requantization implementation with:

```text
-march=rv64imc -mabi=lp64 -ffreestanding -fno-builtin
-nostdlib -nostartfiles
```

The current executable links successfully for the target. Its approximate
layout is 11.9 MiB of read-only model data and 4.9 MiB of static activation
storage; this is why the activation buffers are not stack objects. The build
does not require `pk`.

Inspect the target ELF and verify its sections and ISA:

```sh
riscv64-unknown-elf-size -A build/resnet18_b0.elf
riscv64-unknown-elf-objdump -d build/resnet18_b0.elf \
  > build/resnet18_b0.disassembly
```

The disassembly should contain RV64IMC code and no floating-point or vector
instructions.

### 7.5 Validate the complete B0 model under Spike

The validation build executes the complete 37-operator model before comparing
any tensor. It does not use `pk`, an operating system, a file system, or an
instruction trace. The input and host reference tensors are compiled into
static headers:

```sh
python3 scripts/make_input_header.py \
  model/test_vectors/resnet18/input_0.bin build/resnet18_input.h
python3 scripts/make_tensor_reference_header.py \
  model/test_vectors/resnet18/validation_metadata.json \
  build/resnet18_reference_tensors.h
```

`resnet18_validation_main.c` runs the same executor and compares every
available nonconstant tensor against its compiled-in LiteRT reference after
inference completes. Constant tensors and tensors without preserved host
snapshots are skipped.

Build and run the complete target-side validation with:

```sh
./scripts/validate_resnet18_b0_spike.sh
```

The script compiles `build/resnet18_b0_validation.elf` and runs it under the
modified Spike without `-l` and without `--instructions` by default. Therefore
the entire model is allowed to execute. The timeout is only a safety bound:

```text
SPIKE_INSTRUCTIONS=0
SPIKE_TIMEOUT=1800
```

The limit can be enabled for an intentionally bounded diagnostic run, but it
must not be used for the normal complete-model validation:

```sh
SPIKE_INSTRUCTIONS=5000000000 SPIKE_TIMEOUT=1800 \
  ./scripts/validate_resnet18_b0_spike.sh
```

### 7.6 Fast `tohost` status protocol

The linker provides globally visible `tohost` and `fromhost` symbols in an
aligned `.tohost` section. The startup code clears both symbols. The
validation program uses the Spike HTIF protocol:

- A successful validation performs a proxy `exit(0)` request through `tohost`.
- A failed validation writes a diagnostic line through a proxy `write(1,...)`
  request, then writes a small failure status to `tohost`.
- No `c.ebreak`, PC marker decoding, instruction log, or tensor dump is used.

The failure status is:

```text
tohost = ((2 + tensor_id) << 1) | 1
```

Spike reports the corresponding failure code and the script returns nonzero.
The low payload values are intentionally used here because a raw `tohost = 1`
is interpreted as a proxy syscall request by Spike rather than as a clean
validation exit.

### 7.7 Detailed mismatch diagnostics

On the first mismatching tensor, the target stores:

```c
resnet18_validation_tensor
resnet18_validation_mismatches
resnet18_validation_first_index
resnet18_validation_expected
resnet18_validation_actual
```

It also writes one machine-readable line to Spike stdout:

```text
RESNET18_FAIL tensor=0x00000000 mismatches=0x0000c3fe \
first_index=0x00000000 expected=0xee actual=0x00
```

The validation script preserves this output in:

```text
results/raw/resnet18_b0_validation.stdout
```

and prints it directly on failure. Decode the fields as follows:

1. `tensor` is the first mismatching tensor ID in graph order.
2. `mismatches` is the total number of differing INT8 elements in that tensor.
3. `first_index` is the first differing flattened element index.
4. `expected` is the LiteRT reference INT8 value.
5. `actual` is the RV64 executor INT8 value.

The corresponding stderr file,
`results/raw/resnet18_b0_validation.stderr`, contains Spike's HTIF status
message. The volatile symbols remain available for debugger-based inspection
when a lower-level investigation is needed:

```sh
riscv64-unknown-elf-nm -n build/resnet18_b0_validation.elf | \
  grep resnet18_validation_
```

This workflow completes the full inference first and reports only after the
comparison begins. The current run reports the first mismatch without a trace;
for example, tensor 0 can be investigated immediately from the five fields
above. The same protocol can be reused unchanged for X1 and X2 validation.

Two bugs were fixed during this pass. The PAD implementation now interprets
the FlatBuffer `[4, 2]` padding tensor as
`[batch_before, batch_after, height_before, height_after, width_before,
width_after, channel_before, channel_after]`, rather than treating its flat
storage as four top/bottom/left/right values. The convolution input zero-point
parameter is signed, preventing `-14` from being converted to
`4294967282`. The final fully connected path also applies ReLU only when the
operator metadata requests a fused activation.

## 8. Choose the convolution implementation

The current B0 implementation uses direct scalar convolution. An alternative
implementation to evaluate after correctness is established is `im2col` plus
the existing GEMM path, because it may simplify reuse of `qgemm_scalar()`:

```text
im2col(input) × weight_matrix = output_matrix
```

The reduction length is:

```text
kernel_height * kernel_width * input_channels
```

For later ResNet layers this is usually divisible by eight. The first
convolution and any other non-divisible reduction must use a scalar tail.

The direct kernel avoids materializing an `im2col` buffer, while the GEMM
version may make the later X1/X2 integration simpler. The benchmark must
include whichever transformation and data movement are actually used.

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

## 10. Add B1, X1, and X2 kernels

The ResNet executor uses one compile-time `RESNET18_EXECUTION_MODE` selector,
so every configuration executes the same exported graph, input fixture, tensor
storage, and non-convolution operators:

| Mode | Value | Convolution implementation |
|---|---:|---|
| B0 | 0 | Original scalar convolution and scalar requantization |
| B1 | 1 | Software-optimized scalar convolution with hoisted padding and reduced address recalculation |
| X1 | 2 | Packed `xqmac8` accumulation with scalar requantization |
| X2 | 3 | Packed `xqmac8` accumulation and `xqrequant8` requantization |

B1 is a software-only intermediate for the incremental comparison
`B0 -> B1 -> X1 -> X2`. X1 and X2 share the packed convolution loop. X2 uses
`xqrequant8` for channels whose exported shifts fit `0..63`; the eight channels
requiring wider shifts use the exact scalar wide fallback.

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

For the current X2 baseline, dispatch the eight channels whose exact shift is
outside the `xqrequant8` range to `requantize_scalar_wide()`. A future X2
variant may instead use an expanded `xqrequant8` encoding with a wider shift or
multiplier representation. Treat that as a separate ISA experiment: preserve
the current instruction's numerical behavior, regenerate the encoding and
Spike support, and compare its instruction count and model accuracy against
the baseline X2 path.

The custom paths must produce exactly the same output tensors as B0 for the
same input and model parameters. A scalar tail handles reduction lengths that
are not divisible by eight, including the first convolution's 27-element
reduction.

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

## 12. Build and validate all bare-metal model modes

Build separate configurations so the measured code differs only in the
selected kernel implementation:

```text
build/resnet18_b0.elf
build/resnet18_b1.elf
build/resnet18_x1.elf
build/resnet18_x2.elf
```

Use the mode-aware build helper:

```sh
./scripts/build_resnet18_b0.sh b0
./scripts/build_resnet18_b0.sh b1
./scripts/build_resnet18_b0.sh x1
./scripts/build_resnet18_b0.sh x2
```

The helper passes `-DRESNET18_EXECUTION_MODE=0..3` and keeps
`-march=rv64imc`; no `F`, `D`, or vector extension is added.

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

Run the same input and model under B0, B1, X1, and X2. The bare-metal program
must write a deterministic pass/fail result and enter the normal terminal
loop. The functional validation build uses the `tohost` protocol described in
Section 7; performance-only benchmark harnesses may use their existing
terminal-loop convention.

Use `benchmark_start` and `benchmark_end` symbols around the model inference
call. Exclude startup, reference generation, output comparison, and terminal
loops from kernel-only measurements. Also report full-model counts when
feasible; do not compare mismatched measurement regions.

## 14. Measure the complete model

The complete-model measurement command is:

```sh
./scripts/run_resnet18_benchmarks.sh
```

It builds and runs all four modes with identical static data and input,
captures raw Spike traces and disassemblies, counts the
`benchmark_start`/`benchmark_end` interval, and writes
`results/tables/resnet18_summary.csv`. The exported graph contains exactly
`1,814,073,344` logical MACs. These are ISA-level measurements; Spike does
not provide cycle, CPI, cache, or timing results.

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
./scripts/freeze_resnet18.sh
python3 scripts/inspect_litert_model.py \
    --model model/source/resnet18/resnet18_int8_channelwise.tflite
python3 scripts/export_litert_model.py \
    --model model/source/resnet18/resnet18_int8_channelwise.tflite \
    --out-dir model/exported/resnet18 \
    --test-dir model/test_vectors/resnet18
./scripts/run_resnet18_benchmarks.sh
```

The benchmark script must:

1. Verify the model file and recorded hash.
2. Build B0, B1, X1, and X2.
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
- B0, B1, X1, and X2 execute the same complete graph under Spike.
- X1 and X2 execute the custom instructions in the major dot-product kernels.
- Final outputs match the host reference within the documented integer rule.
- No floating-point extension is required.
- Per-layer and complete-model benchmark results are saved.
- B0/X1/X2 measurements use identical inputs and measurement boundaries.
- README commands reproduce the export, build, verification, and benchmark flow.
