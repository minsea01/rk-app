#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RKNN_ENV="${RKNN_ENV:-/home/minsea/rknn_env}"
PYTHON_BIN="${RKNN_ENV}/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: RKNN python not found: $PYTHON_BIN" >&2
  echo "Hint: set RKNN_ENV or create venv at /home/minsea/rknn_env" >&2
  exit 2
fi

if [[ $# -eq 0 ]]; then
  cat <<'EOF'
Usage:
  scripts/convert_rknn.sh --onnx <model.onnx> --out <model.rknn> [convert args...]

Examples:
  scripts/convert_rknn.sh --onnx artifacts/models/yolo11n.onnx --out /tmp/yolo11n.rknn --no-quant
  RKNN_ENV=/home/minsea/rknn_env scripts/convert_rknn.sh --onnx in.onnx --out out.rknn --calib datasets/calib.txt
EOF
  exit 0
fi

has_reorder_arg=0
for arg in "$@"; do
  if [[ "$arg" == "--reorder" ]] || [[ "$arg" == --reorder=* ]]; then
    has_reorder_arg=1
    break
  fi
done

if [[ "$has_reorder_arg" -eq 0 ]]; then
  set -- "$@" --reorder ""
fi

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  exec "$PYTHON_BIN" "$REPO_ROOT/tools/convert_onnx_to_rknn.py" "$@"
