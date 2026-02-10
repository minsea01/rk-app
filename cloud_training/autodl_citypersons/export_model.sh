#!/usr/bin/env bash
# Deprecated wrapper for tools/export_yolov8_to_onnx.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/lib/deprecation.sh"

warn_deprecated "cloud_training/autodl_citypersons/export_model.sh" "python tools/export_yolov8_to_onnx.py"

MODEL="${MODEL:-outputs/yolov8n_citypersons/weights/best.pt}"
OUTDIR="$(dirname "$MODEL")"
OUTFILE="${OUTFILE:-best.onnx}"

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  exit 1
fi

python "$REPO_ROOT/tools/export_yolov8_to_onnx.py" \
  --weights "$MODEL" \
  --imgsz "${IMG_SIZE:-640}" \
  --opset "${OPSET:-12}" \
  --simplify \
  --outdir "$OUTDIR" \
  --outfile "$OUTFILE"

echo "Exported ONNX model: $OUTDIR/$OUTFILE"

