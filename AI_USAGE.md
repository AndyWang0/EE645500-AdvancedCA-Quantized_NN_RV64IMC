# AI Assistance Disclosure

OpenAI Codex assisted with:

- Reviewing the project requirements, implementation, and benchmark results.
- Implementing shared scalar/custom convolution paths and benchmark-runner
  reliability checks.
- Adding regression tests and organizing synthetic model assets.
- Revising technical documentation and checking measurement consistency.

Verification used host kernel tests, deterministic instruction tests,
tensor-level model validation, Spike profiling, and artifact-hash checks.
Commands, numerical contracts, coverage, and limitations are documented in
[README.md](README.md), [DESIGN.md](DESIGN.md), and
[EXPERIMENTS.md](EXPERIMENTS.md).

AI-assisted changes remain subject to project-author review. This disclosure
does not imply independent verification of model accuracy, hardware
performance, a clean-checkout Spike build, or presentation readiness.
