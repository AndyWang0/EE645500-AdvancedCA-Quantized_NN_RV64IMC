# Custom Instruction Integration in Spike

The custom simulator implements the contract in [DESIGN.md](../../DESIGN.md).
Keep its source/build tree outside this repository. The tracked
[xqmac8.h](xqmac8.h) and [xqrequant8.h](xqrequant8.h) are Spike C++ execution
bodies, not application headers. Applications emit encodings through
[xq_ops.h](../kernels/xq_ops.h).

## Source changes

Starting with an external [upstream Spike checkout](https://github.com/riscv-software-src/riscv-isa-sim),
modify four files:

| Spike-relative path | Change |
|---|---|
| `riscv/encoding.h` | Match/mask constants and instruction declarations |
| `riscv/riscv.mk.in` | Instruction build-list registration |
| `riscv/insns/xqmac8.h` | Packed MAC execution body |
| `riscv/insns/xqrequant8.h` | Requantization execution body |

The recorded custom commit is
`a80897fc5b7b6c491019b58145338e1678a30366`. This identifies the tested
local simulator, not a public upstream base. The exact upstream base,
configure flags, and fresh-checkout reconstruction are not yet verified.

### 1. Register encodings

Add the following definitions inside the constant-definition include guard
of `riscv/encoding.h`:

~~~c
#define MATCH_XQMAC8 0x0000000b
#define MASK_XQMAC8 0xfe00707f
#define MATCH_XQREQUANT8 0x0200000b
#define MASK_XQREQUANT8 0xfe00707f
~~~

Add these entries inside the separate, existing `#ifdef DECLARE_INSN`
section:

~~~c
DECLARE_INSN(xqmac8, MATCH_XQMAC8, MASK_XQMAC8)
DECLARE_INSN(xqrequant8, MATCH_XQREQUANT8, MASK_XQREQUANT8)
~~~

Keep `DECLARE_INSN` at the beginning of its line because build tools parse
these declarations. Do not move the declaration section into the constant
include guard. The masks fix opcode, `funct3`, and `funct7`, leaving
register fields variable; check for local encoding collisions.

Upstream generates `encoding.h` from opcode definitions. Regeneration can
overwrite manual additions; preserve the changes or integrate them into the
generator.

### 2. Install execution bodies

Compare existing same-named files before copying; these commands replace them:

~~~sh
PROJECT_ROOT="/absolute/path/to/EE645500-AdvancedCA-Quantized_NN_RV64IMC"
SPIKE_SOURCE="/absolute/path/to/riscv-isa-sim"
cp "$PROJECT_ROOT/src/isa/xqmac8.h" "$SPIKE_SOURCE/riscv/insns/xqmac8.h"
cp "$PROJECT_ROOT/src/isa/xqrequant8.h" "$SPIKE_SOURCE/riscv/insns/xqrequant8.h"
~~~

Spike includes these bodies in generated execution functions. Do not add
standalone functions or once-only header guards. Both bodies require RV64.
The MAC reads the old destination before modulo-`2^64` accumulation.
Requantization uses host `__int128` arithmetic, ties-away-from-zero
rounding, and sign-extended INT8 saturation. Host wide arithmetic does not
introduce target floating-point instructions.

### 3. Register build entries

In `riscv/riscv.mk.in`, add each instruction once to a list feeding
`riscv_insn_list`. For the recorded layout, insert this after the complete
`riscv_insn_ext_i` assignment and before the aggregate list is consumed:

~~~make
riscv_insn_ext_i += xqmac8 xqrequant8
~~~

Do not insert comments inside backslash-continued assignments or duplicate
existing entries. Encoding declarations alone do not build execution bodies;
generated build files are not the source of registration changes.

This integration enables the custom operations under `--isa=RV64IMC`.
It does not add standard RV64IMC instructions or independently selectable
extension controls. A reusable extension needs explicit feature gating.

## Build

Install the dependencies required by the selected
[upstream revision](https://github.com/riscv-software-src/riscv-isa-sim#build-steps),
including a C++ compiler with `__int128`, Make, Boost, and a device-tree
compiler. A fresh build directory uses:

~~~sh
SPIKE_SOURCE="/absolute/path/to/riscv-isa-sim"
cd "$SPIKE_SOURCE"
mkdir -p build
cd build
../configure
make -j2
./spike -h
~~~

For an already configured checkout:

~~~sh
SPIKE_SOURCE="/absolute/path/to/riscv-isa-sim"
make -C "$SPIKE_SOURCE/build" -j2
~~~

Reconfigure in a new build directory if generated instruction files remain
stale. No system-wide installation is required. `SPIKE_BIN` can select an
executable in any build directory; the legacy `SPIKE_DIR` fallback assumes
`build/spike`.

## Verification

Run from the workload repository root, replacing the custom executable path
in every command:

~~~sh
# Specify the custom Spike executable.
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/test_custom_spike.sh
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/test_synthetic_qgemm.sh
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/run_synthetic_benchmarks.sh
~~~

The instruction test must report 67 MAC and 73 requant operations. The small
runners check HTIF completion, process status, trace coverage, and exact
custom counts; a matching help banner is insufficient verification.
[The synthetic workflow](../../docs/synthetic_workflow.md) specifies the
protocol. Full-model preparation, validation, and profiling are documented in
[README.md](../../README.md).

GNU assembler emits `.insn` without mnemonic support. GNU disassembly may
show raw words; the analyzers decode custom encodings directly. Additional
Spike disassembler integration may be needed for new mnemonics.

## Further instructions and reproducibility

For another instruction, define numerical/exception/overlap behavior, allocate
a non-conflicting encoding, register its build entry, implement its body,
and compare directed/randomized execution with an independent reference
before integrating kernels.

A guide and execution bodies avoid distributing the large simulator tree,
but do not establish a reproduced upstream build. Record a retrievable base
revision and tested build flags. A reachable custom-source commit or small
patch can supplement this recipe. Local comments, paths, and inaccessible
commit hashes are not substitutes for distributed source changes.
