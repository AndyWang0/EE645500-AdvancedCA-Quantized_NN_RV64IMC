# Quantized Neural-Network ISA Extension

## 1. Scope and Compatibility Contract

This document defines two RV64 custom instructions for quantized INT8 neural-network kernels:

- `xqmac8`: computes an eight-element signed INT8 dot product and adds it to a 64-bit accumulator.
- `xqrequant8`: applies a project-defined fixed-point scale, rounding, output-zero-point addition, and signed INT8 saturation to a 64-bit accumulator.

Both instructions are register-only. They do not access memory, modify source registers, or set condition flags. The instructions do not raise arithmetic exceptions for any source-register values. Unallocated instruction encodings must be treated as illegal instructions by the implementation.

The two instructions use the R-type encoding and the RISC-V `custom-0` major opcode (`0x0b`, binary `0001011`). The instruction semantics below are the architectural contract; the Spike implementation, C reference, test generator, and inline-assembly wrappers must all follow it exactly.

### 1.1 Quantization compatibility

`xqrequant8` uses the explicitly defined one-rounding rule in Section 4: round `A * M / 2^S` to nearest, with exact ties away from zero. This is a project-defined fixed-point format. It is **not automatically bit-identical to an unmodified LiteRT/TensorFlow Lite kernel**, whose multiplier encoding and rounding implementation can differ.

For this project-defined format, the Python exporter and C reference must produce and consume the `M`, `S`, and `Z` values defined here. If bit-exact compatibility with a particular LiteRT/TFLite runtime is required, the instruction's requantization semantics must instead be specified to match that runtime's exact multiplier, shift, and rounding implementation; do not claim bit-exact compatibility based only on approximate scale equivalence.

### 1.2 Supported affine-quantized dot products

`xqmac8` computes a raw dot product of signed INT8 values. It does not apply activation or weight zero points itself.

This design supports the common case where weights are symmetric-quantized with `weight_zero_point = 0`, while activations may have a nonzero signed INT8 zero point `z_a`. For a reduction over `K` elements:

`sum((q_a - z_a) * q_w) = sum(q_a * q_w) - z_a * sum(q_w)`.

Therefore, initialize the accumulator with the bias correction:

`acc_init = bias - z_a * sum(q_w)`

and then accumulate the raw dot product using `xqmac8`. Weight sums may be precomputed per output channel. This equation assumes `weight_zero_point = 0`. Nonzero weight zero points require additional correction terms and are outside this design's direct-kernel contract unless software implements those terms separately.

## 2. Instruction Encoding

The standard R-type instruction fields are:

| Field | Width | Description |
|---|---:|---|
| `funct7` | 7 bits | Selects the custom operation |
| `rs2` | 5 bits | Second source register |
| `rs1` | 5 bits | First source register |
| `funct3` | 3 bits | Operation subfield |
| `rd` | 5 bits | Destination register |
| `opcode` | 7 bits | `0001011` (`custom-0`, `0x0b`) |

The instruction assignments are:

| Instruction | `funct7` | `funct3` | Opcode | Match value | Mask |
|---|---|---|---|---|---|
| `xqmac8` | `0000000` | `000` | `0x0b` | `0x0000000b` | `0xfe00707f` |
| `xqrequant8` | `0000001` | `000` | `0x0b` | `0x0200000b` | `0xfe00707f` |

For the R-type mask, opcode, `funct3`, and `funct7` are fixed; register fields are not part of the match. The encodings are distinct. The implementation must reject unallocated custom instruction encodings according to its illegal-instruction policy.

## 3. `xqmac8`: Eight-Way INT8 Multiply-Accumulate

### 3.1 Assembly syntax

```asm
xqmac8 rd, rs1, rs2
```

### 3.2 Operand interpretation

Each 64-bit source register contains eight packed signed INT8 values:

| Source | Bits | Meaning |
|---|---|---|
| `rs1` | `[8*i+7 : 8*i]` | Activation-like operand `A[i]` |
| `rs2` | `[8*i+7 : 8*i]` | Weight-like operand `W[i]` |

for `i = 0..7`. Byte zero is in bits `[7:0]`; byte order within each packed register is defined by bit position, with element zero in the least-significant byte. Each byte is interpreted as signed two's-complement INT8. The instruction does not load bytes from memory and has no independent memory-endianness behavior.

`rd` is both the accumulator input and destination. This read-modify-write behavior is intentional and must be visible to software constraints and, in any future hardware implementation, to dependency/hazard logic. Register overlap is legal. All operands are sampled from their pre-instruction values before the destination is written.

### 3.3 Architectural semantics

Let `old_rd` be the 64-bit bit pattern read from `rd`. Sign-extend each byte of `rs1` and `rs2` to a signed integer, then compute the dot product:

```text
dot = 0
for i = 0 .. 7:
    a = sign_extend_i8(rs1[8*i+7 : 8*i])
    w = sign_extend_i8(rs2[8*i+7 : 8*i])
    dot = dot + a * w

rd = (old_rd + dot) modulo 2^64
```

The dot-product sum itself is safely representable in a signed 32-bit temporary (its range is within `[-131072, 131072]`); the accumulator update is modulo `2^64`. Interpret the final 64-bit pattern as a signed two's-complement accumulator when used by later arithmetic instructions.

Implementations in C/C++ must not rely on signed-overflow undefined behavior for the final addition. A reference can perform the addition in `uint64_t` and preserve the resulting 64-bit pattern.

### 3.4 Intended use

The instruction performs eight signed INT8 products and accumulates their sum in one architectural instruction:

```text
acc = acc_init
for each packed activation/weight pair:
    xqmac8 acc, activation_pack, weight_pack
```

The accumulation dependency remains serial when each instruction updates the same accumulator register. This design reduces architectural instruction count; it does not, by itself, promise an eight-fold cycle-speedup. Dynamic instruction count must be measured in Spike, and cycle claims require a suitable cycle-accurate model.

### 3.5 Affine activation zero-point correction

For symmetric weights (`weight_zero_point = 0`) and activation zero point `z_a`, initialize `rd` to:

```text
acc_init = bias - z_a * sum(weight_q[k])
```

where `weight_q[k]` is the signed quantized weight and the sum covers the full reduction dimension for one output channel. The correction must be included once per output accumulator, not once per packed group. Then `xqmac8` accumulates `sum(activation_q[k] * weight_q[k])` in groups of eight.

Handle a reduction tail whose length is not divisible by eight with a scalar reference loop or another explicitly verified tail path. Do not read beyond the tensor boundary merely to fill a packed word.

## 4. `xqrequant8`: Fixed-Point Requantization and Saturation

### 4.1 Assembly syntax

```asm
xqrequant8 rd, rs1, rs2
```

### 4.2 Operand interpretation

`rs1` contains the signed 64-bit accumulator `A`. `rs2` contains a packed parameter word:

| Bits | Width | Field | Interpretation |
|---|---:|---|---|
| `[31:0]` | 32 | `M` | Signed two's-complement multiplier |
| `[37:32]` | 6 | `S` | Unsigned right-shift amount, `0..63` |
| `[45:38]` | 8 | `Z` | Signed two's-complement output zero point |
| `[63:46]` | 18 | Reserved | Software must write zero; hardware ignores these bits |

`M` and `Z` are sign-extended from their declared widths before arithmetic. Neural-network scale factors are nonnegative, so the model exporter is expected to generate `M >= 0`; negative `M` nevertheless has well-defined arithmetic semantics. The output zero point `Z` must be representable as signed INT8.

The parameter register is explicit and has no persistent side effects. Different output channels may use different parameter words.

### 4.3 Fixed-point scale representation

This instruction defines the approximation:

`real_multiplier ~= M / 2^S`.

For a real quantization multiplier `r > 0`, the Python exporter may choose `S` in `0..63` and set `M = round_nearest_ties_away_from_zero(r * 2^S)`, subject to `1 <= M <= 2^31 - 1`. Prefer the largest `S` that keeps `M` representable to maximize precision. If no nonzero representable pair exists, the exporter must report that the scale is unsupported rather than silently wrapping or clamping it. For `r == 0`, use `M = 0, S = 0`.

For a standard affine quantized operator, a typical real multiplier is `r = input_scale * weight_scale[channel] / output_scale`. Weight scales may be per-channel, so the `M` and `S` values can differ by output channel. The exact exporter rounding convention must match the reference implementation.

### 4.4 Architectural semantics

Use signed 128-bit intermediate arithmetic for the multiply, rounding, shift, and zero-point addition. With a signed 64-bit `A` and signed 32-bit `M`, the product fits in a signed 128-bit integer.

```text
P = signed(A) * signed(M)

if S == 0:
    scaled = P
else:
    magnitude = abs(P)
    rounded_magnitude = (magnitude + 2^(S-1)) >> S
    if P < 0:
        scaled = -rounded_magnitude
    else:
        scaled = rounded_magnitude

q = scaled + signed(Z)

if q > 127:
    result = 127
else if q < -128:
    result = -128
else:
    result = q

rd = sign_extend_i8(result)
```

Rounding is round-to-nearest with exact ties away from zero. For `S = 0`, no rounding or right shift is performed. The result is clamped after adding `Z`, then sign-extended to 64 bits. The output is a signed INT8 quantized value in an RV64 register; storing it in a tensor requires an ordinary store or an explicit packing operation in software.

All source values are read before `rd` is written. Register overlap, including `rd = rs1` or `rd = rs2`, is legal. Reserved bits in `rs2[63:46]` are ignored by the instruction; well-formed software must set them to zero for reproducibility and forward compatibility.

### 4.5 Intended use

```text
xqrequant8 output_q, accumulator, quant_params
```

This replaces the project's defined fixed-point multiply, rounding, shift, output-zero-point addition, and INT8 clamp. It does not include bias addition or activation zero-point correction; those are handled in the accumulation path.

## 5. Compiler/Assembler Interface

GCC does not need to recognize the mnemonics for the initial implementation. GNU assembler's `.insn` directive can emit the custom encodings. The wrapper must accurately describe read/write operands to the compiler.

Example wrapper for `xqmac8`:

```c
static inline int64_t xqmac8(int64_t acc,
                            uint64_t packed_a,
                            uint64_t packed_w)
{
    __asm__ volatile (
        ".insn r 0x0b, 0x0, 0x00, %0, %1, %2"
        : "+r" (acc)
        : "r" (packed_a), "r" (packed_w)
    );
    return acc;
}
```

Example wrapper for `xqrequant8`:

```c
static inline int64_t xqrequant8(int64_t acc, uint64_t params)
{
    int64_t out;
    __asm__ volatile (
        ".insn r 0x0b, 0x0, 0x01, %0, %1, %2"
        : "=r" (out)
        : "r" (acc), "r" (params)
    );
    return out;
}
```

These wrappers assume that the encoding table in Section 2 has been implemented identically in Spike. Validate the emitted instruction words with `riscv64-unknown-elf-objdump -d`. If the assembler or compiler version rejects `.insn`, use an explicitly encoded `.word` only after independently checking the bit fields; do not guess the encoding.

## 6. Reference Model and Verification

The reference model is normative: C/Python reference behavior, Spike execution, and emitted machine code must agree bit-for-bit for the defined integer operations.

### 6.1 `xqmac8` directed tests

- All activation and weight bytes zero.
- Eight `1 * 1` products with an initial accumulator of `10` (expected result `18`).
- Mixed positive and negative operands.
- Byte values `127` and `-128`.
- Positive and negative initial accumulators.
- Repeated accumulation across multiple packed groups.
- `rd` overlapping with `rs1` or `rs2`.
- Accumulator wraparound near the signed 64-bit limits, checked using modulo-`2^64` reference arithmetic.

### 6.2 `xqrequant8` directed tests

- `M = 0` and `S = 0`.
- `S = 0`, `1`, and `63`.
- Positive and negative products.
- Positive and negative exact rounding ties.
- Values just below and above rounding ties.
- Output saturation below `-128` and above `127`.
- `Z = -128`, `0`, and `127`.
- `rd` overlapping with `rs1` or `rs2`.
- Nonzero reserved bits, which must not change the result.

### 6.3 Randomized tests

Generate deterministic random inputs with a recorded seed. Compare Spike outputs with an independent reference implementation for thousands of cases per instruction. Use host-side signed 128-bit arithmetic or Python arbitrary-precision integers for the requantization reference; do not compute `A * M` in signed 64-bit arithmetic and allow it to overflow.

For a model operator, compare every output element against the same project-defined quantized reference, not only a checksum. Also test model dimensions whose reduction length is and is not divisible by eight.

## 7. Benchmark Methodology

Run at least three configurations using identical tensors and parameters:

- `B0`: scalar MAC and scalar project-defined requantization.
- `X1`: `xqmac8` with scalar project-defined requantization.
- `X2`: `xqmac8` plus `xqrequant8`.

Measure at least one dot-product microbenchmark, two major model operators, and one complete model block or full inference if available. Save raw Spike traces, disassembly, exact commands, compiler flags, tool versions, tensor dimensions, random seed, and output verification results.

Report:

- Total retired/dynamic instructions in the stated counting scope.
- Load, store, scalar multiply, branch, integer arithmetic, and custom-instruction counts.
- `xqmac8` and `xqrequant8` dynamic counts and frequencies.
- Logical MACs per retired instruction.
- Dynamic instructions per output element.
- Static code size.
- Instruction-count reduction from `B0` to `X1` and from `X1` to `X2`.

Spike instruction counts are ISA-level measurements. They are not, by themselves, cycle counts, CPI, or evidence of a cycle-accurate speedup.

## 8. Design Tradeoffs and Limitations

### 8.1 Accumulator width and dependency

A 64-bit register accumulator avoids a hidden accumulator state and accommodates common INT8 reductions. The `xqmac8` destination is also an implicit input, so compiler wrappers must use a read/write constraint and any future hardware implementation must treat `rd` as a source for hazard/dependency purposes. The result uses modulo-`2^64` arithmetic; workloads requiring an exact mathematical sum must ensure that the intended sum is in range.

### 8.2 Requantization parameters and precision

Packing `M`, `S`, and `Z` in `rs2` makes the instruction stateless and allows per-channel values without CSR writes. The cost is that software must prepare or reuse the parameter word. The representable scale is limited by the signed 32-bit multiplier and 6-bit shift. The exporter must detect unsupported scales rather than silently changing them.

### 8.3 Separation of MAC and requantization

Keeping accumulation and requantization separate preserves precision across the dot-product loop and allows the effect of each instruction to be measured independently. The `xqmac8` instruction reduces arithmetic instruction count but does not remove loads, pointer arithmetic, loop control, bias correction, or reduction tails. Those costs must remain in the benchmark.

### 8.4 Runtime compatibility

The project-defined requantization rule is deterministic and straightforward to verify, but it must not be described as bit-identical to LiteRT/TFLite without a direct comparison against the exact target runtime. Likewise, the raw MAC instruction requires the affine zero-point correction described in Section 1.2 for asymmetric activations.
