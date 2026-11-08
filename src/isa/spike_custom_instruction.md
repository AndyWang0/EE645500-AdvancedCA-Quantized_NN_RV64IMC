# Adding the project's custom instructions to Spike

Keep the Spike source checkout and its build directory outside this workload
repository. Only the instruction bodies and this integration guide are stored
here; a complete Spike source tree or binary is not necessary.

The architectural contract is [DESIGN.md](../../DESIGN.md). The files
[xqmac8.h](xqmac8.h) and [xqrequant8.h](xqrequant8.h) are **Spike C++ instruction
bodies**, not application inline-assembly headers. Applications emit the
instructions through [xq_ops.h](../kernels/xq_ops.h).

## 1. Start from an upstream Spike checkout

Use your own external checkout of
[upstream Spike](https://github.com/riscv-software-src/riscv-isa-sim).
Starting from unmodified source, apply the additions in Sections 2–4 to these
four files:

| File relative to the Spike checkout | Purpose |
|---|---|
| `riscv/encoding.h` | Match/mask constants and `DECLARE_INSN` registrations |
| `riscv/riscv.mk.in` | Include both instructions in the simulator build |
| `riscv/insns/xqmac8.h` | Packed signed INT8 MAC execution semantics |
| `riscv/insns/xqrequant8.h` | Fixed-point rounding and saturation execution semantics |

The encoding and build additions are provided below, and both instruction bodies are included in this repository.

The custom branch/revision recorded in the README is historical provenance,
not an upstream checkout target. The exact tested upstream base revision still
needs to be recorded, and this recipe needs a fresh-checkout build trial.

## 2. Define the instruction encodings

In `riscv/encoding.h`, add these constants alongside the existing match/mask
definitions, inside the definitions include guard:

```c
#define MATCH_XQMAC8 0x0000000b
#define MASK_XQMAC8 0xfe00707f
#define MATCH_XQREQUANT8 0x0200000b
#define MASK_XQREQUANT8 0xfe00707f
```

In the same file, add the following lines **inside the existing
`#ifdef DECLARE_INSN` section**, alongside the other `DECLARE_INSN` entries:

```c
DECLARE_INSN(xqmac8, MATCH_XQMAC8, MASK_XQMAC8)
DECLARE_INSN(xqrequant8, MATCH_XQREQUANT8, MASK_XQREQUANT8)
```

Keep `DECLARE_INSN` at the start of its line: Spike build machinery can parse
these entries as text. The declaration section is intentionally separate from
the guarded constant definitions; do not move it inside that definitions guard.

| Instruction | Opcode | `funct3` | `funct7` | Match | Mask |
|---|---|---|---|---|---|
| `xqmac8` | `0x0b` (`custom-0`) | `0` | `0x00` | `0x0000000b` | `0xfe00707f` |
| `xqrequant8` | `0x0b` (`custom-0`) | `0` | `0x01` | `0x0200000b` | `0xfe00707f` |

These masks constrain opcode, `funct3`, and `funct7`, leaving register fields
variable. Check for collisions with other locally added instructions.
`encoding.h` is generated upstream, so regenerating it from `riscv-opcodes`
can overwrite manual changes. Preserve/reapply these additions, or integrate
the encodings into that generator for longer-term maintenance.

## 3. Install the instruction bodies

Copy the two small, tracked source files from this repository. These commands
replace same-named files in the external checkout: compare any existing local
versions before running them.

```sh
# Replace both paths; do not point PROJECT_ROOT at the Spike checkout.
PROJECT_ROOT="/absolute/path/to/EE645500-AdvancedCA-Quantized_NN_RV64IMC"
SPIKE_SOURCE="/absolute/path/to/riscv-isa-sim"
cp "$PROJECT_ROOT/src/isa/xqmac8.h" "$SPIKE_SOURCE/riscv/insns/xqmac8.h"
cp "$PROJECT_ROOT/src/isa/xqrequant8.h" "$SPIKE_SOURCE/riscv/insns/xqrequant8.h"
```

Each body is included by Spike's generated execution functions; do not wrap it
in a standalone function or add a conventional once-only header guard.

- `xqmac8` requires RV64, sign-extends eight byte lanes from each source,
  sums their products, and adds the sum to the old destination modulo `2^64`.
  Destination register `rd` is also an input. Source/destination overlap must
  observe all old operand values before writeback.
- `xqrequant8` requires RV64 and uses a signed 128-bit intermediate product
  for signed 64-bit accumulator times signed 32-bit multiplier. It rounds ties
  away from zero, adds the signed INT8 zero point, clamps to `[-128, 127]`,
  and writes a sign-extended result. The host C++ compiler must support
  `__int128`; this does not introduce floating-point instructions in the target.

The exact arithmetic is in the linked bodies and [DESIGN.md](../../DESIGN.md).
The six-bit shift field represents `0..63`; software must use the existing
wide scalar fallback for unrepresentable model parameters.

## 4. Register the instructions in the build

In `riscv/riscv.mk.in`, add each name once to an instruction list that feeds
`riscv_insn_list`. For this simple project integration, one option is to insert
the following **after the complete `riscv_insn_ext_i` assignment and before the
aggregate instruction list is consumed**:

```make
riscv_insn_ext_i += xqmac8 xqrequant8
```

Do not insert a comment inside a backslash-continued assignment. If the names
are already listed in your checkout, do not add them again. Follow
the surrounding structure if using a different Spike revision.

This makes Spike generate and register both instruction implementations.
An encoding declaration alone is insufficient if its instruction is omitted
from the build. Do not manually edit generated build files as the source fix.

This project enables the instructions in its custom simulator while invoking
`--isa=RV64IMC`; it does not define a separately selectable extension flag.
That is convenient for the experiment, but is not a standard RV64IMC feature.
A reusable extension would additionally need explicit enable/disable checks
and ISA-extension registration, outside this four-file recipe.

## 5. Configure and rebuild externally

Follow the dependency instructions for your selected
[upstream Spike revision](https://github.com/riscv-software-src/riscv-isa-sim#build-steps).
Typical Linux requirements include a C++ compiler, Make, Boost development
libraries, and the device-tree compiler. For a fresh, unconfigured build:

```sh
SPIKE_SOURCE="/absolute/path/to/riscv-isa-sim"
cd "$SPIKE_SOURCE"
mkdir -p build
cd build
../configure
make -j2
./spike -h
```

No system-wide installation is needed: use the resulting `build/spike` directly.
For an already configured checkout, run:

```sh
SPIKE_SOURCE="/absolute/path/to/riscv-isa-sim"
make -C "$SPIKE_SOURCE/build" -j2
```

If generated instruction files are stale after changing the registration or
build list, reconfigure/rebuild in a new build directory. `SPIKE_BIN` can select
the executable from any build directory; only the `SPIKE_DIR` fallback assumes
the directory is named `build`.

## 6. Select the correct binary and verify

Run workload commands from this project's root. Replace the absolute paths in
every example; prefer the explicit executable path:

| Variable | Meaning | Used by |
|---|---|---|
| `SPIKE_DIR` | Compatibility fallback checkout containing `build/spike`, used only when `SPIKE_BIN` is unset | Instruction tests, GEMM tests, synthetic benchmark |
| `SPIKE_BIN` | Full custom executable path | All validation, test, and benchmark runners |
| `SPIKE_SOURCE` | External source Git checkout | Model freezing/provenance, together with `SPIKE_BIN` |

First run the small instruction and GEMM tests:

```sh
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/test_custom_spike.sh
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/test_synthetic_qgemm.sh
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/run_synthetic_benchmarks.sh
```

The documented instruction-test result is
`instruction_test passed (67 xqmac8, 73 xqrequant8)`.
The small runners require explicit HTIF success, exit status zero, complete
trace markers, and exact custom counts. They use a wall-clock timeout instead
of an instruction budget and publish nothing from a failed invocation. See
[the synthetic workflow](../../docs/synthetic_workflow.md). A matching
help/version banner alone does not demonstrate that the instructions work.

After preparing/exporting the model as in [README.md](../../README.md), run:

```sh
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/validate_resnet18_b0_spike.sh b0
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/validate_resnet18_b0_spike.sh b1
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/validate_resnet18_b0_spike.sh x1
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/validate_resnet18_b0_spike.sh x2
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/run_resnet18_benchmarks.sh
```

An unmodified GNU assembler need not recognize these mnemonics: the workload
uses `.insn` in [xq_ops.h](../kernels/xq_ops.h). GNU `objdump` may print raw
instruction words; the project's histogram analyzer recognizes the encodings.
If a named disassembly is needed for a new instruction, Spike disassembler
formatting may need additional integration; the four-file recipe does not
promise arbitrary new mnemonics will be formatted as desired.

## 7. Extending the recipe to another instruction

1. Define operands, widths, signedness, overflow, rounding, exceptions, and
   overlap behavior before assigning a non-conflicting encoding.
2. Add match/mask constants and a `DECLARE_INSN` entry.
3. Implement `riscv/insns/<name>.h` and register the name in the build list.
4. Rebuild, then emit the encoding with `.insn` and compare directed/randomized
   cases against an independent software reference.
5. Integrate the kernel and measure instruction counts. Add simulator extension
   gating or disassembler support if the instruction requires it.

See upstream's [new-instruction instructions](https://github.com/riscv-software-src/riscv-isa-sim#simulating-a-new-instruction).
Some upstream prose references `opcodes.h`; this project uses `encoding.h`.
The relevant implementation structure can be checked in upstream
[encoding.h](https://github.com/riscv-software-src/riscv-isa-sim/blob/master/riscv/encoding.h)
and [riscv.mk.in](https://github.com/riscv-software-src/riscv-isa-sim/blob/master/riscv/riscv.mk.in),
but these moving links are not a pinned experimental base.

## Reproducibility limits of a guide-only approach

Keeping a large Spike build outside the repository is appropriate. This guide
and the two instruction-body files provide a small reconstruction recipe.
However, comments do not distribute missing source changes, and a local path or
custom commit hash is unusable to another person unless the corresponding
source repository is accessible. The upstream base revision is not established
by the four paths alone.

For stronger reproducibility without including Spike, record the exact tested
upstream base and any configure flags, plus either a reachable custom-source
repository/commit or a small four-file patch. A patch is optional here and much
smaller than a source/build tree. Until a fresh-checkout reconstruction is
tested, treat the guide as instructions, not evidence of a reproduced build.
