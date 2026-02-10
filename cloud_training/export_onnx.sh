#!/usr/bin/env bash
# Deprecated wrapper for tools/export_yolov8_to_onnx.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/lib/deprecation.sh"

warn_deprecated "cloud_training/export_onnx.sh" "python tools/export_yolov8_to_onnx.py"

WORK_DIR="${WORK_DIR:-$HOME/pedestrian_training}"
BEST_PT="${BEST_PT:-$WORK_DIR/outputs/yolov8n_pedestrian/weights/best.pt}"
OUTDIR="${OUTDIR:-$WORK_DIR/outputs/yolov8n_pedestrian/weights}"

if [[ ! -f "$BEST_PT" ]]; then
  echo "Model not found: $BEST_PT" >&2
  exit 1
fi

python "$REPO_ROOT/tools/export_yolov8_to_onnx.py" \
  --weights "$BEST_PT" \
  --imgsz 640 \
  --opset 12 \
  --simplify \
  --outdir "$OUTDIR" \
  --outfile best_640.onnx

python "$REPO_ROOT/tools/export_yolov8_to_onnx.py" \
  --weights "$BEST_PT" \
  --imgsz 416 \
  --opset 12 \
  --simplify \
  --outdir "$OUTDIR" \
  --outfile best_416.onnx

echo "Exported ONNX models to: $OUTDIR"

