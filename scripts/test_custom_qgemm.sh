#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Compatibility alias; the canonical test checks BOTH X1 and X2.
exec bash "$ROOT_DIR/scripts/test_synthetic_qgemm.sh" "$@"
