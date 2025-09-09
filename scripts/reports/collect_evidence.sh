#!/usr/bin/env bash
set -euo pipefail

# Collects training metrics, conversion logs, and any provided device logs
# Usage: scripts/collect_evidence.sh <RUN_NAME>

RUN_NAME=${1:-exp}
OUT_DIR=artifacts
LOG_DIR=$OUT_DIR/logs
REP_DIR=$OUT_DIR/reports
MODEL_DIR=$OUT_DIR/models

mkdir -p "$LOG_DIR/net" "$LOG_DIR/cam" "$LOG_DIR/fps" "$REP_DIR" "$MODEL_DIR"

echo "[INFO] Collecting training artifacts for run: $RUN_NAME"
if [ -d "runs/train/$RUN_NAME" ]; then
  rsync -a --include 'results.csv' --include 'confusion_matrix.png' --include 'PR_curve.png' \
    --include 'F1_curve.png' --include 'P_curve.png' --include 'R_curve.png' \
    --exclude '*' "runs/train/$RUN_NAME/" "$LOG_DIR/train_$RUN_NAME/" || true
else
  echo "[WARN] runs/train/$RUN_NAME not found"
fi

echo "[INFO] Capturing model artifacts"
for f in artifacts/models/*.onnx artifacts/models/*.rknn; do
  [ -e "$f" ] && cp -f "$f" "$MODEL_DIR/" || true
done

# Device logs are expected to be copied into artifacts/logs/net|cam|fps beforehand
echo "[INFO] Generating summary report"
SUMMARY="$REP_DIR/summary.md"
{
  echo "# Project Evidence Summary"
  echo
  echo "## Training"
  echo "- Run: $RUN_NAME"
  echo "- Metrics: logs/train_$RUN_NAME/results.csv (if present)"
  echo
  echo "## Models"
  ls -1 "$MODEL_DIR" 2>/dev/null || echo "(no models copied)"
  echo
  echo "## Network Throughput"
  ls -1 "$LOG_DIR/net" 2>/dev/null || echo "(put iperf3 logs here: eth0_up.txt, eth0_down.txt, eth1_up.txt, eth1_down.txt, dual.txt)"
  echo
  echo "## Camera Bench"
  ls -1 "$LOG_DIR/cam" 2>/dev/null || echo "(put camera fps logs here)"
  echo
  echo "## Inference FPS"
  ls -1 "$LOG_DIR/fps" 2>/dev/null || echo "(put device inference FPS logs here)"
} > "$SUMMARY"

echo "[DONE] Summary generated at $SUMMARY"

