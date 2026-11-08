# ResNet-18 Model Preparation and Execution

This workflow converts a fixed quantized LiteRT graph into static C data for
bare-metal RV64IMC execution. It does not implement a general-purpose runtime.
Dependency and top-level execution commands are in [README.md](../README.md).

## Frozen workload

The selected artifact is `resnet18_int8_channelwise.tflite` from
`litert-community/resnet18`. It uses signed INT8 activations with asymmetric
quantization and signed INT8 weights with symmetric channelwise quantization.
The floating-point and weight-only variants are not the target workload.

| Artifact | SHA-256 |
|---|---|
| Model FlatBuffer | `8fea1acb62829e120b4953b6fbf0f842e0032909254f51c2d989703b66534683` |
| Upstream model card | `b7f5d7d636ff1f50b6b1b4bf870e143e67d563ab3352666a867a04d950180d68` |

[freeze_resnet18.sh](../scripts/freeze_resnet18.sh) verifies these hashes before
replacing downloads and records source/tool metadata in
[model_metadata.json](../model/source/resnet18/model_metadata.json).
The recorded remote revision is `main`; hashes identify the exact contents.
The upstream [model card](../model/source/resnet18/README.md) is preserved
unchanged as provenance. Model and generated-weight redistribution remains
subject to the source licensing terms.

~~~sh
# Specify the external custom Spike executable and source checkout.
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" \
SPIKE_SOURCE="/absolute/path/to/riscv-isa-sim" ./scripts/freeze_resnet18.sh
~~~

The model FlatBuffer is ignored by Git. A clean checkout requires acquisition
and export before a full-model build. The initial acquisition environment is
recorded in the metadata; the subsequently resolved host environment is
recorded separately in
[host_requirements.txt](../model/source/resnet18/host_requirements.txt).

## Graph inspection

Use the project virtual environment prepared as described in
[README.md](../README.md):

~~~sh
PYTHONPATH="$PWD" .venv/bin/python scripts/inspect_litert_model.py \
    --model model/source/resnet18/resnet18_int8_channelwise.tflite \
    --json model/source/resnet18/graph_report.json \
    --summary model/source/resnet18/graph_summary.txt
~~~

The [graph report](../model/source/resnet18/graph_report.json) contains tensor
types/shapes, scales, zero points, constants, operator order, strides,
dilations, padding, and fused activations. Inspection uses the FlatBuffer
operator list rather than host-added delegate nodes.

The graph has one subgraph, 86 tensors, and 37 operators:

| Operator | Count | Executor behavior |
|---|---:|---|
| TRANSPOSE | 1 | NCHW input to NHWC |
| PAD | 5 | Pad with the tensor's quantized zero point |
| CONV_2D | 20 | Channelwise weights; scalar or packed reduction |
| MAX_POOL_2D | 1 | Integer maximum |
| ADD | 8 | Quantized residual addition and optional fused activation |
| AVERAGE_POOL_2D | 1 | Centered integer averaging |
| FULLY_CONNECTED | 1 | Scalar reduction and mode-selected requantization |

| Tensor | Shape | Scale | Zero point |
|---|---|---:|---:|
| Input 85 | `[1,3,224,224]`, NCHW | 0.01865844801068306 | -14 |
| Output 41 | `[1,1000]` | 0.1705460548400879 | -61 |

Quantized values represent `scale * (q - zero_point)`. Convolution weights
use `[output_channel, kernel_height, kernel_width, input_channel]` layout.
The first convolution has shape `[64,7,7,3]`; its 147-element reduction
requires three scalar tail elements after packed groups of eight.

## Host reference and static export

Generate the deterministic input and preserve host tensor snapshots:

~~~sh
PYTHONPATH="$PWD" .venv/bin/python scripts/validate_litert_model.py \
    --model model/source/resnet18/resnet18_int8_channelwise.tflite \
    --out-dir model/test_vectors/resnet18 \
    --debug-dir model/test_vectors/host_validation/intermediate
~~~

The default seed is `20261104`. The script writes input/output binaries,
validation metadata, and available intermediate snapshots. These are numerical
test fixtures, not an ImageNet accuracy evaluation.

Export the graph and constants:

~~~sh
PYTHONPATH="$PWD" .venv/bin/python scripts/export_litert_model.py \
    --model model/source/resnet18/resnet18_int8_channelwise.tflite \
    --out-dir model/exported/resnet18 \
    --test-dir model/test_vectors/resnet18
~~~

Generated artifacts include:

| Artifact | Contents |
|---|---|
| `resnet18_graph.h/.c` | Tensor declarations, shared storage, graph tables, operator options |
| `resnet18_weights.h`, `resnet18_biases.h` | Typed constant arrays |
| `resnet18_shapes.h` | Tensor dimensions |
| `resnet18_quant_params.h` | Scales, zero points, weight sums, fixed-point/fallback parameters |
| `model_metadata.json` | Complete exported tensor/operator records |

Generated ResNet headers and graph source are ignored by Git. The compact
test fixtures remain under [model/test_vectors/resnet18/](../model/test_vectors/resnet18/).
The exporter can copy existing input/output binaries from the sibling
`host_validation/` directory; it does not generate host validation metadata
or intermediate snapshots. Fixtures must retain the same model, seed, and
reference metadata throughout export and validation.

Synthetic JSON, headers, and vectors are separate under the corresponding
`synthetic/` directories; [export_synthetic_model.py](../scripts/export_synthetic_model.py)
does not parse LiteRT models.

## Numerical contract

The target implements affine dot-product correction:

~~~text
acc = bias + sum(input_q * weight_q)
           - input_zp * weight_sum
           - weight_zp * input_sum
           + K * input_zp * weight_zp
~~~

All selected weight zero points are zero. Weight sums are exported per output
channel as INT32 values using INT64 host accumulation.

Real channel scales are converted to integer multiplier/shift pairs.
[DESIGN.md](../DESIGN.md) specifies the project-defined output rounding:
nearest with ties away from zero, followed by output-zero-point addition
and INT8 saturation. Agreement with LiteRT is verified for the frozen fixture;
the rule is not a universal LiteRT rounding guarantee.

First-convolution channels `[2,4,7,9,13,36,38,48]` need shifts beyond the
custom instruction's 0–63 range. The exporter records wide fallback pairs
(up to shift 126) and dispatch flags. Software preserves those parameters
rather than clamping them to the custom encoding.

Residual ADD combines its scaled inputs before the final rounding and
saturation. Fused ReLU clamps to the output zero point. Global average pooling
uses centered integer accumulation and ties-away-from-zero division; the
selected pooling input/output share scale and zero point.

## Static executor

The [executor](../src/benchmarks/resnet18_b0_executor.c) dispatches operators
in exported order. Constants are read-only; nonconstant tensors have static
storage owned by the generated graph source. There is no runtime FlatBuffer
parser, dynamic allocation, or tensor-sized stack allocation. Activation
buffers are not yet reused according to tensor lifetimes.

B0 uses direct scalar convolution. B1/X1/X2 reuse one contiguous input patch
across output channels and hoist channel corrections. X1/X2 pack activations
once per output position and weights within the reduction loop. X2 selects
custom requantization where parameters fit; other channels retain the wide
scalar path. Fully connected MACs remain scalar. Scratch limits, mode values,
and reentrancy constraints are specified in [DESIGN.md](../DESIGN.md).

Padding reads the FlatBuffer `[4,2]` before/after pairs in dimension order.
Signed zero points remain signed during arithmetic. Activation fusion follows
the operator metadata rather than an unconditional ReLU assumption.

## Validation and profiling

A native B0 comparison isolates graph and arithmetic errors from target
execution:

~~~sh
./scripts/build_resnet18_b0_host_dump.sh
.venv/bin/python scripts/compare_resnet18.py \
    --host-dir model/test_vectors/resnet18 \
    --ported-dir build/resnet18_b0_tensors
~~~

The comparator checks tensors marked available in both dump sets. It reports
element counts, maximum absolute error, and the first mismatch. Unavailable
tensors are not evidence of a successful comparison.

Target validation builds the same executor with compiled-in reference tensors:

~~~sh
# Specify the custom Spike executable; repeat with b1, x1, and x2.
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/validate_resnet18_b0_spike.sh b0
~~~

The validator completes inference before comparing available nonconstant
tensors. It exits through HTIF; failure reports tensor ID, mismatch count,
first index, and expected/actual bytes in `RESNET18_FAIL` output.
The default wall-clock bound is 1,800 seconds. Keep
`SPIKE_INSTRUCTIONS=0`; an instruction-limited diagnostic run is not a
complete validation result.

Build and profiling entry points are documented in [README.md](../README.md).
Profiling uses PC histograms rather than unbounded instruction traces.
[EXPERIMENTS.md](../EXPERIMENTS.md) distinguishes whole-ELF counts from
function-self rows and reports convolution, fully connected, and complete
inference results. Profiling does not replace tensor validation.
