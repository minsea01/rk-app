#!/usr/bin/env bash
# Deprecated wrapper for tools/export_yolov8_to_onnx.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/lib/deprecation.sh"

warn_deprecated "tools/export.sh" "python tools/export_yolov8_to_onnx.py"

MODEL_PATH="${1:-best.pt}"
IMG_SIZE="${IMG_SIZE:-640}"
OPSET="${OPSET:-12}"
OUTDIR="${OUTDIR:-artifacts/models}"
OUTFILE="${OUTFILE:-best_static640.onnx}"

python "$REPO_ROOT/tools/export_yolov8_to_onnx.py" \
  --weights "$MODEL_PATH" \
  --imgsz "$IMG_SIZE" \
  --opset "$OPSET" \
  --simplify \
  --outdir "$OUTDIR" \
  --outfile "$OUTFILE"

