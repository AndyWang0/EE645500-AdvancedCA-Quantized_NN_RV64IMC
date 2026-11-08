#!/usr/bin/env bash
set -euo pipefail

# Use the same fresh-artifact and mode checks as the selective runner.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT_DIR/scripts/run_resnet18_benchmarks_selective.sh" "$@"
