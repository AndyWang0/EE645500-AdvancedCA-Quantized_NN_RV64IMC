#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Compatibility alias; use run_synthetic_benchmarks.sh for new workflows.
exec bash "$ROOT_DIR/scripts/run_synthetic_benchmarks.sh" "$@"
