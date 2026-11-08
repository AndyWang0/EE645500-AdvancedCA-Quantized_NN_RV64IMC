# ISA-Level Evaluation

## Workload and environment

The workload is the frozen W8A8 ResNet-18 graph: 37 operators,
1,814,073,344 logical MACs, and 1,000 final output elements. All modes use
identical weights, quantization parameters, and deterministic input seed
`20261104`. Model identity and operator inventory are recorded in
[model provenance](model/source/resnet18/model_metadata.json) and
[graph report](model/source/resnet18/graph_report.json).

Measurements use Python 3.10.12, RISC-V GCC 10.2.0, and custom Spike revision
`a80897fc5b7b6c491019b58145338e1678a30366`.
[Host requirements](model/source/resnet18/host_requirements.txt) record the
resolved Python environment. The target uses RV64IMC, LP64, `-O3`,
freestanding compilation, and no floating-point/vector extensions.

B0 is scalar. B1 adds software-only convolution optimization. X1 replaces
the shared reduction with packed MACs. X2 adds custom requantization where
representable. Fully connected MACs and convolution tails remain scalar.
[DESIGN.md](DESIGN.md) defines the instruction and quantization contracts.

## Methodology

Correctness validation and profiling are separate gates. All four model modes
passed comparison of available nonconstant tensors with the preserved host
LiteRT reference. Instruction tests passed 67 MAC and 73 requant cases;
synthetic X1/X2 matched Python-generated output bytes. The parser/runner
regression suite contains 19 tests.

Complete-model measurements use Spike PC histograms (`-g`) matched to the
same ELF's disassembly. Custom encodings are decoded from raw words.
[analyze_histogram.py](scripts/analyze_histogram.py) uses explicit alias-aware
categories: loads, stores, multiply, branch/jump, custom, and residual
`integer` instructions. Categories are mutually exclusive; `integer`
also includes remaining decoded operations such as CSR access and nops.

| Scope | Included execution | Limitation |
|---|---|---|
| `elf_text` | Executed ELF PCs, including input setup, inference, checksum, and termination | Not a temporal slice between benchmark labels; reset ROM is excluded |
| `function_self` | PCs within the named function and compiler clones | Excludes called helpers; not inclusive operator latency/cost |
| `kernel_window` | Synthetic trace from `benchmark_start` to, but not including, `benchmark_end` | Excludes verification, checksum, and HTIF exit |

Memory instructions are loads plus stores, not transferred bytes or cache
misses. MAC/instruction divides logical MACs by instructions in the stated
scope. Whole-model instructions/output uses the 1,000 final logits, not the
number of intermediate activations. Static code size is the entire ELF's
`.text`, including linked functions that may not execute.

## Complete-model results

Source: [resnet18_summary.csv](results/tables/resnet18_summary.csv).

| Mode | Dynamic instructions | MAC/instruction | Reduction vs B0 | `.text` bytes |
|---|---:|---:|---:|---:|
| B0 | 25,847,519,995 | 0.070184 | 0.0000% | 9,178 |
| B1 | 8,987,379,995 | 0.201847 | 65.2292% | 11,234 |
| X1 | 8,609,609,995 | 0.210703 | 66.6908% | 11,436 |
| X2 | 8,504,784,995 | 0.213300 | 67.0963% | 11,526 |

### Instruction mix and utilization

| Mode | Loads | Stores | Integer | Multiply | Branch/jump | `xqmac8` | `xqrequant8` |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 3,591,784,764 | 20,214,736 | 18,132,181,433 | 1,828,001,559 | 2,275,337,503 | 0 | 0 |
| B1 | 3,660,116,121 | 8,983,515 | 3,189,761,316 | 1,821,459,845 | 307,059,198 | 0 | 0 |
| X1 | 2,087,534,597 | 10,884,945 | 5,965,569,152 | 10,306,949 | 308,920,240 | 226,394,112 | 0 |
| X2 | 2,087,534,597 | 10,884,945 | 5,889,432,066 | 5,538,229 | 282,616,686 | 226,394,112 | 2,384,360 |

| Mode | Load/store instructions per MAC | Custom instructions / total |
|---|---:|---:|
| B0 | 1.991099 | 0.0000% |
| B1 | 2.022575 | 0.0000% |
| X1 | 1.156745 | 2.6296% |
| X2 | 1.156745 | 2.6900% |

B0/B1 execute no custom instructions, although their ELFs contain unused
custom GEMM functions. X1/X2 execute exactly
`sum(conv_output_elements * floor(K/8)) = 226,394,112` packed MACs.
This is not the entire model's MAC count divided by eight: tails and the
fully connected reduction remain scalar.

### Major operators

Sources: [B0](results/tables/resnet18_b0.csv),
[B1](results/tables/resnet18_b1.csv), [X1](results/tables/resnet18_x1.csv),
and [X2](results/tables/resnet18_x2.csv). All rows use `function_self`.

| Mode | Convolution self instructions | Fully connected self instructions |
|---|---:|---:|
| B0 | 25,585,775,139 | 5,156,055 |
| B1 | 8,725,276,617 | 5,156,055 |
| X1 | 8,347,508,463 | 5,156,055 |
| X2 | 8,347,508,463 | 5,156,055 |

Convolution accounts for 1,813,561,344 logical MACs and 2,483,712 outputs;
the fully connected layer accounts for 512,000 MACs and 1,000 outputs.
X1/X2 self counts are identical because requantization executes in called
helpers outside the measured function ranges. Their whole-program totals
capture that difference. These rows aggregate all calls to each function;
they do not provide individual-layer inclusive profiles.

## Incremental effects and bottlenecks

B0 → B1 reduces instructions by 65.2292% through patch reuse, hoisted
corrections, and unrolled scalar reduction. B1 → X1 saves 377,770,000
instructions (4.2033%); X1 → X2 saves 104,825,000 (1.2175%).
The large software gain must be distinguished from the incremental ISA gain.

The principal measured overheads are:

1. **Runtime packing and address manipulation.** The dominant eight-product
   loop retires 38 instructions in B1 versus 36 in X1 over 226,394,112
   iterations. X1 eliminates 1,811,152,896 scalar multiplies but adds
   2,775,807,836 residual integer instructions. Constructing packed weights
   requires byte loads, shifts, ORs, and indexing. Offline aligned weight
   packing is the primary optimization candidate.
2. **Data movement.** X2 still executes 2,098,419,542 loads/stores, 24.6734%
   of its instruction count, or 1.156745 per MAC. Packed arithmetic alone
   does not eliminate these instructions. This is instruction evidence,
   not a measurement of memory bandwidth or cache stalls.
3. **Control and output processing.** X2 retains 282,616,686 branch/jump
   instructions. Its 2,384,360 custom requantizations save approximately
   43.96 instructions each relative to X1, but affect far fewer events than
   the packed-MAC loop. Eight first-convolution channels require wide scalar
   fallback; scalar tails and other graph operators also remain.

These overheads overlap and are not independent cycle/stall contributions.
Additional packed layouts and loop restructuring require new correctness and
measurement runs.

## Synthetic GEMM

The fixture uses M=8, N=16, K=32: 4,096 logical MACs and 128 checked outputs.
Source: [synthetic summary](results/tables/synthetic/summary.csv).

| Mode | Kernel instructions | Instructions/output | MAC/instruction | `xqmac8` | `xqrequant8` |
|---|---:|---:|---:|---:|---:|
| B0 | 62,003 | 484.3984 | 0.066061 | 0 | 0 |
| X1 | 54,053 | 422.2891 | 0.075777 | 512 | 0 |
| X2 | 39,537 | 308.8828 | 0.103599 | 512 | 128 |

All modes have 3,332 bytes of linked `.text`. Results use the exported JSON
input and `kernel_window` scope. [Legacy tables](results/tables/synthetic/legacy/)
use a different input and an inclusive end marker; they are not comparable
as a kernel-optimization experiment.

## Reproduction and evidence

[README.md](README.md) provides dependency, model, validation, and benchmark
commands. Profiling commands require an explicit custom Spike executable:

~~~sh
# Specify the custom Spike executable.
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/run_resnet18_benchmarks.sh
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/run_synthetic_benchmarks.sh
~~~

Model runs retain histogram/disassembly pairs and per-mode manifests;
synthetic runs retain trace/disassembly pairs and manifests. Selected results
are staged and published only after all selected runs pass. Build-only model
runs preserve measured artifacts. New raw files are ignored by Git and need
a submission archive if required as evidence. Existing-binary runs and
artifact hashes were verified; reconstruction of Spike from a fresh upstream
checkout remains unverified.

## Performance targets and limits

The evaluated coverage includes a microbenchmark, convolution and fully
connected operators, and complete inference. The specified performance target
is not yet demonstrated: the synthetic B0-to-X2 instruction ratio is about
1.57, fully connected self counts are unchanged, and no cycle measurements
are available. The approximately 3.04-fold whole-program instruction reduction
does not establish a major-GEMM/FC target or an end-to-end cycle speedup.
Runtime packing and data-movement counts quantify the limiting overhead.

Validation establishes agreement for the recorded deterministic fixture,
not ImageNet accuracy or exhaustive input equivalence. Spike provides no
cycle, CPI, cache-miss, energy, Fmax, area, or PPA results. Those require
CV-Wally or another suitable hardware evaluation.
