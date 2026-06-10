#!/usr/bin/env bash
# CI redundancy guardrails.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[redundancy-check] verifying configs symlink..."
test -L configs

echo "[redundancy-check] verifying unified training entrypoint..."
if rg -n "yolo detect train" scripts --glob '*.sh' --glob '!scripts/check_redundancy.sh' | rg -v '^scripts/train\.sh:'; then
  echo "Found direct training loops outside scripts/train.sh" >&2
  exit 1
fi

echo "[redundancy-check] verifying decode_predictions dedup..."
decode_defs="$(rg -n '^def decode_predictions\(' apps | wc -l | tr -d ' ')"
if [[ "$decode_defs" -gt 3 ]]; then
  echo "Expected <= 3 decode_predictions definitions, found: $decode_defs" >&2
  exit 1
fi

echo "[redundancy-check] verifying compatibility wrapper emits deprecation warnings..."
if ! rg -q "warn_deprecated|\\[DEPRECATED\\]" scripts/compare_onnx_rknn.py; then
  echo "Missing deprecation warning in wrapper: scripts/compare_onnx_rknn.py" >&2
  exit 1
fi

echo "[redundancy-check] all checks passed"
