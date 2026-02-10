#!/usr/bin/env bash
# CI redundancy guardrails.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[redundancy-check] verifying configs symlink..."
test -L configs

echo "[redundancy-check] verifying unified training entrypoint..."
if rg -n "yolo detect train" cloud_training scripts --glob '*.sh' --glob '!scripts/check_redundancy.sh' | rg -v '^scripts/train\.sh:'; then
  echo "Found direct training loops outside scripts/train.sh" >&2
  exit 1
fi

echo "[redundancy-check] verifying decode_predictions dedup..."
decode_defs="$(rg -n '^def decode_predictions\(' apps | wc -l | tr -d ' ')"
if [[ "$decode_defs" -gt 3 ]]; then
  echo "Expected <= 3 decode_predictions definitions, found: $decode_defs" >&2
  exit 1
fi

echo "[redundancy-check] verifying wrappers emit deprecation warnings..."
for wrapper in \
  cloud_training/train_runner.sh \
  tools/train_yolov8.py \
  tools/export_rknn.py \
  tools/pc_compare.py \
  scripts/validate_models.py \
  scripts/compare_onnx_rknn.py \
  tools/export.sh \
  cloud_training/export_onnx.sh \
  cloud_training/autodl_citypersons/export_model.sh
do
  if ! rg -q "warn_deprecated|\\[DEPRECATED\\]" "$wrapper"; then
    echo "Missing deprecation warning in wrapper: $wrapper" >&2
    exit 1
  fi
done

echo "[redundancy-check] all checks passed"
