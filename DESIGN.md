# ISA and Software Design

## Scope

The design extends bare-metal RV64IMC with two register-only instructions for
W8A8 inference. No floating-point or vector extension is required. The
instruction bodies are [xqmac8.h](src/isa/xqmac8.h) and
[xqrequant8.h](src/isa/xqrequant8.h); application wrappers are in
[xq_ops.h](src/kernels/xq_ops.h).

The numerical contract below is normative. Spike execution, software
references, and emitted encodings must agree. The instructions do not access
memory, set flags, or raise arithmetic exceptions. Normal RISC-V `x0`
read/write behavior applies. Source/destination overlap is legal: all inputs
are sampled before destination writeback.

## Encodings

Both instructions use the R-type format and the `custom-0` opcode.

| Instruction | Opcode | `funct3` | `funct7` | Match | Mask |
|---|---|---|---|---|---|
| `xqmac8` | `0x0b` | `0` | `0x00` | `0x0000000b` | `0xfe00707f` |
| `xqrequant8` | `0x0b` | `0` | `0x01` | `0x0200000b` | `0xfe00707f` |

The mask fixes opcode, `funct3`, and `funct7`; register fields remain
variable. Unallocated encodings are illegal. The custom Spike build enables
these operations under `--isa=RV64IMC`; it does not provide independent
extension enable/disable controls. These are not standard RV64IMC operations.

## Packed multiply-accumulate: `xqmac8`

~~~asm
xqmac8 rd, rs1, rs2
~~~

Each source contains eight two's-complement INT8 lanes. Lane `i` occupies
bits `[8*i+7:8*i]`, with lane zero in the least-significant byte.
`rd` is both a 64-bit accumulator input and the destination.

~~~text
dot = sum(sign_extend_i8(rs1[8*i+7:8*i])
        * sign_extend_i8(rs2[8*i+7:8*i]), i = 0..7)
rd = (old_rd + dot) modulo 2^64
~~~

The eight-product sum fits INT32, with range `[-130048, 131072]`.
The final addition wraps modulo `2^64`; references use unsigned arithmetic
to avoid signed-overflow undefined behavior. Packed registers are defined by
bit position, independently of memory endianness.

The instruction replaces scalar product/reduction work but retains operand
loads, packing, address generation, and loop control. A dependency through
the accumulator remains between successive instructions.

## Requantization: `xqrequant8`

~~~asm
xqrequant8 rd, rs1, rs2
~~~

`rs1` holds a signed INT64 accumulator `A`; `rs2` holds:

| Bits | Field | Interpretation |
|---|---|---|
| `[31:0]` | `M` | Signed INT32 multiplier |
| `[37:32]` | `S` | Unsigned right shift, 0–63 |
| `[45:38]` | `Z` | Signed INT8 output zero point |
| `[63:46]` | Reserved | Ignored by execution; software writes zero |

The instruction approximates a real scale by `M / 2^S`. All signed
multipliers, including negative values, have defined semantics. Model scales
are nonnegative.

~~~text
P = signed(A) * signed(M)
if S == 0:
    scaled = P
else:
    magnitude = (abs(P) + 2^(S-1)) >> S
    scaled = -magnitude if P < 0 else magnitude
rd = sign_extend_i8(clamp(scaled + signed(Z), -128, 127))
~~~

Multiplication, rounding, and zero-point addition use signed 128-bit
intermediates. Rounding is nearest with exact ties away from zero.
Saturation follows zero-point addition. The instruction does not add bias,
apply input zero-point correction, or store the output.

This single-rounding rule is project-defined. It is not a general claim of
bit-exact compatibility with every LiteRT/TFLite requantization kernel.
Compatibility is established by tensor comparison for the frozen model and
fixture.

## Quantized operator mapping

For affine-quantized operands:

~~~text
acc = bias + sum(input_q * weight_q)
           - input_zp * sum(weight_q)
           - weight_zp * sum(input_q)
           + K * input_zp * weight_zp
~~~

The selected model has symmetric weights (`weight_zp=0`). Exported
per-channel weight sums permit initialization with
`bias - input_zp * weight_sum`, followed by raw packed products.
Padding uses the activation zero point. Incomplete groups of eight use a
scalar tail without reading beyond tensor boundaries.

The [LiteRT exporter](scripts/export_litert_model.py) normalizes real scales
with `frexp`, rounds the significand using Python `round`, and emits a
signed INT32 multiplier with a power-of-two shift. This parameter-conversion
step is distinct from the instruction's ties-away-from-zero output rounding.
Eight channels of the first convolution need shifts beyond 63; exported
wide parameters and `requantize_scalar_wide` preserve the scalar result.
Unsupported parameters are not clamped to the custom field range.

## Software configurations

All modes execute the same static graph, tensors, and non-convolution
operators in the [executor](src/benchmarks/resnet18_b0_executor.c).

| Mode | Selector | Convolution | Requantization |
|---|---:|---|---|
| B0 | 0 | Direct scalar reduction | Scalar |
| B1 | 1 | Reusable input patch; eight-element unrolled scalar reduction | Scalar |
| X1 | 2 | B1 preparation/traversal with `xqmac8` | Scalar |
| X2 | 3 | X1 reduction | `xqrequant8` where representable; wide scalar fallback otherwise |

B1/X1/X2 share an always-inlined convolution core behind noinline entry
points. A single-position contiguous patch is reused across output channels.
Channel correction terms are hoisted. X1/X2 pack activations once per output
position; weights are packed at runtime. Scratch storage is at most 15,360
bytes, with reduction capacity 4,608 and channel capacity 512. Larger shapes
fall back to B0. Static scratch makes this implementation non-reentrant.

Fully connected MACs and convolution tails remain scalar in every model mode.
X2 also accelerates eligible fully connected requantization. The independent
synthetic GEMM exercises packed MACs in both X1 and X2.

## Compiler interface and verification

GNU assembler emits the operations through `.insn`; no compiler opcode
extension is required. The MAC wrapper uses `"+r"` for its accumulator.
The requant wrapper uses the accumulator as both input and output and encodes
`rd=rs1`. Other overlap combinations are legal architectural cases, not
an exhaustive test-coverage claim.

The [instruction test](tests/test_custom_spike.c) compares execution with
independent integer references:

| Instruction | Directed cases | Random cases | Seed |
|---|---:|---:|---|
| `xqmac8` | 3 | 64 | `0x2d5a5a1234567890` |
| `xqrequant8` | 9 | 64 | `0x9b1d4e6f10293847` |

The tests exercise signed lanes, large/negative accumulators, wraparound,
rounding, saturation, and shifts through 63. Exhaustive register-overlap and
reserved-bit tests are not implemented. Synthetic outputs are checked against
Python-generated references; full-model validation compares available
nonconstant tensors with preserved host LiteRT outputs.

## Evaluation and limitations

[EXPERIMENTS.md](EXPERIMENTS.md) defines measurement scopes and reports the
incremental comparisons B0 → B1 → X1 → X2. Patch gathering, correction,
packing, and tail work remain part of execution.

The ISA reduces arithmetic instructions but does not remove memory traffic
or packing. Per-channel explicit parameters avoid hidden state but require
parameter preparation. A 64-bit accumulator avoids special accumulator
registers but introduces a read-modify-write dependency. RTL hazard logic,
latency, area, and critical-path cost require separate hardware evaluation.
