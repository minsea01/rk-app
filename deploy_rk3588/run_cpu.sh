#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=${1:-"${DIR}/best.onnx"}
SRC=${2:-"${DIR}/../industrial_dataset/images/val"}

echo "Using model: $MODEL"
echo "Source: $SRC"

yolo predict model="$MODEL" source="$SRC" imgsz=640 device=cpu conf=0.45 iou=0.50 agnostic_nms=True save=True name=cpu_pred || {
  echo "If onnxruntime not available, try .pt model:"
  echo "  yolo predict model=${DIR}/../runs/detect/industrial_y8s_stable/weights/best.pt source=$SRC imgsz=640 device=cpu save=True"
}


