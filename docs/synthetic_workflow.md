# Synthetic GEMM and Instruction Tests

The synthetic workload verifies the custom instructions independently of the
ResNet-18 graph. The shared harness is
[synthetic_gemm_main.c](../src/benchmarks/synthetic_gemm_main.c); arithmetic
is implemented in [qgemm.c](../src/kernels/qgemm.c).
The fixture has M=8, N=16, K=32, 4,096 logical MACs, and 128 output elements.

## Assets and entry points

| Location | Contents |
|---|---|
| [model/source/synthetic/](../model/source/synthetic/) | Synthetic JSON model |
| [model/exported/synthetic/](../model/exported/synthetic/) | Weights, quantization, metadata, input, and expected-output C headers |
| [model/test_vectors/synthetic/](../model/test_vectors/synthetic/) | Matching binary fixtures and metadata |
| [build/synthetic/](../build/synthetic/) | ELF files and staged runs |
| [results/raw/synthetic/](../results/raw/synthetic/) | Traces, disassemblies, logs, and manifests |
| [results/tables/synthetic/](../results/tables/synthetic/) | Per-mode CSVs and selected-mode summary |

[export_synthetic_model.py](../scripts/export_synthetic_model.py) computes
expected outputs using Python integer arithmetic. It accepts synthetic JSON,
not LiteRT/TFLite. Each mode compares every output byte with this reference.
Before a GEMM run, the runner re-exports into staging and rejects missing or
stale project headers/vectors without modifying them.

~~~sh
python3 scripts/export_synthetic_model.py
# Specify the custom Spike executable in each invocation.
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/test_custom_spike.sh
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/test_synthetic_qgemm.sh
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/run_synthetic_benchmarks.sh
~~~

The GEMM test runs both X1 and X2 using distinct test artifact names.
Optional benchmark selection:

~~~sh
# Specify the custom Spike executable; B0 comparisons are N/A unless selected.
SPIKE_BIN="/absolute/path/to/riscv-isa-sim/build/spike" ./scripts/run_synthetic_benchmarks.sh --modes x1 x2
~~~

All entry points accept `--output-root DIRECTORY` to isolate build/results;
sources and fixtures still come from the current checkout. `SPIKE_BIN` takes
precedence over the `SPIKE_DIR/build/spike` compatibility fallback.
`SPIKE_TIMEOUT` is a finite positive wall-clock bound, default 30 seconds.
No instruction budget is used; nonzero `SPIKE_INSTRUCTIONS` is rejected.

| Compatibility entry point | Behavior |
|---|---|
| [export_model.py](../scripts/export_model.py) | Synthetic exporter with synthetic-only defaults |
| [run_benchmarks.sh](../scripts/run_benchmarks.sh) | Delegates to the synthetic benchmark runner |
| [test_custom_qgemm.sh](../scripts/test_custom_qgemm.sh) | Tests both custom GEMM modes |
| [build_b0.sh](../scripts/build_b0.sh) | Builds and measures synthetic B0; replaces the summary with the selected B0 row |

Previous generic asset/output paths are not aliases. Full-model paths are
unchanged. Synthetic source selection is `BENCHMARK_MODE=0/1/2` for
B0/X1/X2; it is distinct from the four-mode ResNet selector.

## Completion and publication

[baremetal_status.h](../src/benchmarks/baremetal_status.h) writes
`EE645500_PASS <workload>` or `EE645500_FAIL <workload>` through HTIF,
then requests exit. Nops during HTIF handling are not success evidence.
[run_synthetic.py](../scripts/run_synthetic.py) requires:

- Exit status zero, exactly one matching pass line, and no failure line.
- A fresh, nonempty, newline-terminated trace with valid single-hart records.
- Executed PCs present in the paired disassembly, except the reset ROM.
- Exactly one executed `workload_exit` marker.
- Exact custom counts and, for GEMM, one complete ordered kernel window.

| Workload | `xqmac8` | `xqrequant8` |
|---|---:|---:|
| Instruction directed/randomized tests | 67 | 73 |
| Synthetic B0 | 0 | 0 |
| Synthetic X1 | 512 | 0 |
| Synthetic X2 | 512 | 128 |

Simulator failures and invalid output return nonzero; timeout returns 124.
Published measured artifacts remain unchanged until every selected workload
passes. Failed/staged logs remain in the printed run directory.
Manifests record arguments and source/artifact hashes. Summaries include only
selected modes and never read a stale B0 comparison.

## Accounting and regression coverage

[analyze_trace.py](../scripts/analyze_trace.py) shares classification with the
histogram analyzer and decodes custom instructions by instruction word.
The scope is `[benchmark_start, benchmark_end)`, including the kernel call
and helpers but excluding verification, checksum, and HTIF exit. Static
`.text` covers the entire linked ELF.

Current results and performance limits are in [EXPERIMENTS.md](../EXPERIMENTS.md).
[Legacy tables](../results/tables/synthetic/legacy/) used a different runtime
input and included the end-marker instruction; they are not directly
comparable with the JSON-input, half-open-window results.

~~~sh
python3 -m unittest discover -s tests -p 'test_*.py' -v
~~~

The suite contains 12 [synthetic workflow tests](../tests/test_synthetic_workflow.py)
and seven [histogram tests](../tests/test_histogram_analysis.py).
Coverage includes nonzero exits despite pass messages, timeout, missing/fail
messages, stale exports/logs, missing/empty/truncated traces, marker errors,
alias/custom decoding, and preservation of published results after a later
mode fails. The publication integration test requires the cross compiler;
real Spike execution is a separate verification gate.
