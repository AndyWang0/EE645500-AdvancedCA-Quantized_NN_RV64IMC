#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Historical name: this builds AND measures synthetic B0, not ResNet-18.
exec python3 "$ROOT_DIR/scripts/run_synthetic.py" benchmark --modes b0 "$@"
