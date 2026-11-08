# Synthetic GEMM and instruction-test workflow

This small workload is a verification/profiling fixture, not the ResNet-18
model. Its shared C harness is
[synthetic_gemm_main.c](../src/benchmarks/synthetic_gemm_main.c); the instruction
test remains [test_custom_spike.c](../tests/test_custom_spike.c). The full-model
executor and its measurements are unchanged by this cleanup.

## Asset and entry-point migration

| Previous location/name | Current location/name |
|---|---|
| `model/source/model_int8.json` | [model/source/synthetic/model_int8.json](../model/source/synthetic/model_int8.json) |
| `model/exported/{weights,quant_params,model_metadata}.h` | [model/exported/synthetic/](../model/exported/synthetic/) |
| `model/test_vectors/{input.bin,reference_output.bin,metadata.json}` | [model/test_vectors/synthetic/](../model/test_vectors/synthetic/) |
| `scripts/export_model.py` | [export_synthetic_model.py](../scripts/export_synthetic_model.py); old entry point remains a compatibility wrapper |
| `src/benchmarks/{b0_main,x1_main,benchmark_main}.c` | One [synthetic_gemm_main.c](../src/benchmarks/synthetic_gemm_main.c), selected with `BENCHMARK_MODE=0/1/2` |
| `scripts/run_benchmarks.sh` | [run_synthetic_benchmarks.sh](../scripts/run_synthetic_benchmarks.sh); old name delegates to it |
| `scripts/test_custom_qgemm.sh` | [test_synthetic_qgemm.sh](../scripts/test_synthetic_qgemm.sh); old name delegates to it |
| `build/{b0,x1,x2}.elf` | [build/synthetic/](../build/synthetic/) |
| `results/raw/<synthetic-name>.*` | [results/raw/synthetic/](../results/raw/synthetic/) |
| `results/tables/{b0,x1,x2,summary}.csv` | Current [results/tables/synthetic/](../results/tables/synthetic/); old CSVs archived under [legacy/](../results/tables/synthetic/legacy/) |

The misleading old X1 test actually called X2. The replacement tests **both**
X1 and X2 and uses distinct `qgemm_x1_test`/`qgemm_x2_test` artifact names, so
tests cannot overwrite benchmark ELF/trace pairs. The historical
[build_b0.sh](../scripts/build_b0.sh) still builds **and measures** B0; it now
delegates to the shared runner and writes a selected-only B0 summary in the
new location. It is not the full-model build command. Old source/header/output
paths are not aliases; scripts and includes now use the explicit synthetic
locations. Existing old raw traces are left as historical files, not consumed
by the new scripts. ResNet paths and ignore rules have not changed.

## Commands

From the repository root, regenerate the synthetic JSON-derived artifacts:

```sh
python3 scripts/export_synthetic_model.py
```

This writes matching headers (including the new input/reference
[test_vectors.h](../model/exported/synthetic/test_vectors.h)) and binary
fixtures. Python integer arithmetic independently computes the expected
outputs. Defaults are synthetic-only; this script does **not** parse TFLite.
Use [export_litert_model.py](../scripts/export_litert_model.py) for ResNet-18.
The runner re-exports to its staging directory and rejects stale/missing
committed headers or binary vectors without rewriting them.

Replace the custom Spike executable path in **each** invocation:

```sh
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/test_custom_spike.sh
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/test_synthetic_qgemm.sh
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/run_synthetic_benchmarks.sh
# Optional selected modes; comparison with B0 is N/A when B0 is not selected.
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/run_synthetic_benchmarks.sh --modes x1 x2
```

`SPIKE_BIN` takes precedence over the legacy `SPIKE_DIR/build/spike` fallback.
`SPIKE_TIMEOUT` is a finite positive wall-clock limit in seconds (default 30).
No `--instructions` budget is used; a nonzero `SPIKE_INSTRUCTIONS` is rejected.
An instruction budget can cause Spike to return zero before the workload
finishes, so zero alone is insufficient success evidence.

All three entry points accept `--output-root DIRECTORY` to isolate their
build/results from normal published files. Sources and fixtures still come
from this checkout. The shared implementation is
[run_synthetic.py](../scripts/run_synthetic.py).

## Completion, failure, and publication

[baremetal_status.h](../src/benchmarks/baremetal_status.h) writes an exact
`EE645500_PASS <workload>` or `EE645500_FAIL <workload>` line using HTIF,
then exits via `tohost` (zero for success, nonzero for failure). No proxy
kernel is required. Nops during asynchronous HTIF handling do not signal pass.

The runner requires **all** of:

- Simulator exit zero, exactly one matching pass line, and no failure line.
- A new nonempty, newline-terminated trace in a unique staging directory.
- Valid single-hart trace records and mapped ELF PCs (except the reset ROM).
- Exactly one executed `workload_exit` marker after HTIF message completion.
- Exact dynamic custom counts: instruction test 67/73; synthetic B0 0/0,
  X1 512/0, X2 512/128 for MAC/requant respectively.
- For GEMM, one complete ordered kernel window and identical custom counts
  inside that window. Missing/repeated markers or missing kernel PCs fail.

A crash/illegal instruction preserves its nonzero status; timeout returns
124; output/trace/count failures return nonzero. All logs remain in the
printed staging directory for diagnosis. No published measured output is
updated until **every selected workload** passes. Each completed workload
gets a JSON run manifest with tool arguments, source and artifact hashes.
The summary uses only modes from that invocation, never an old B0 row.
Staging directories are retained for diagnosis; they may be removed manually
when no longer needed.

## Measurement and verification record

The trace analyzer shares alias/category classification with
[analyze_histogram.py](../scripts/analyze_histogram.py) and decodes custom
instructions from raw words, even when a disassembler prints `unknown`.
Synthetic counts cover `[benchmark_start, benchmark_end)`: argument setup
inside the markers, the call, and all called kernel helpers; not correctness
checking, checksum, or HTIF. This differs from full-model `elf_text` counts.
`.text` reports the entire synthetic ELF, including other linked functions.

Fresh runs with GCC 10.2.0 and the external custom Spike at recorded revision
`a80897fc5b7b6c491019b58145338e1678a30366` passed:

| Workload | Dynamic kernel instructions | `xqmac8` | `xqrequant8` |
|---|---:|---:|---:|
| Instruction directed/randomized tests | Not a timed kernel | 67 | 73 |
| Synthetic B0 | 62,003 | 0 | 0 |
| Synthetic X1 | 54,053 | 512 | 0 |
| Synthetic X2 | 39,537 | 512 | 128 |

The fixture is M=8, N=16, K=32: 4,096 logical MACs and 128 checked outputs.
The old harness generated a **different** input from its exported JSON and
compared only to the C scalar kernel. The new one uses the JSON input and
checks every output against Python. Together with the corrected half-open
window, this starts a new synthetic baseline: do not describe the difference
from legacy counts as a kernel optimization. GEMM arithmetic was not changed.

Focused automated regressions:

```sh
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

[test_synthetic_workflow.py](../tests/test_synthetic_workflow.py) checks
nonzero exits even with a pass message, timeout, no pass, explicit failure,
missing/empty/truncated traces, stale logs/exports, nops without completion,
marker ordering/repetition, raw custom decoding, alias categories, and
preservation of published artifacts when a later selected mode fails.
The existing [histogram tests](../tests/test_histogram_analysis.py) remain in
the same regression suite. Cross-compiler-dependent publication tests skip
when the toolchain is unavailable; real Spike execution is a separate gate.

The completed local checks passed all 19 Python tests, the host scalar/custom
GEMM regression, Bash syntax checks, and whitespace checks. Real Spike runs
also verified the compatibility entry points, `SPIKE_DIR` fallback,
`SPIKE_BIN` precedence, the B0-only historical script, and a selected X1/X2
summary with no stale B0 comparison. Published synthetic artifact hashes
matched all six run manifests. All four full-model modes rebuilt successfully;
their ELF, executor, analyzer, graph, histogram, and disassembly hashes still
matched the prior full-model manifests. Full-model simulation was not repeated.
