#!/bin/bash
set -euo pipefail

# RKNN FPS proof (实干版)
# Requires ENABLE_RKNN build and a valid RKNN model configured in config/detection/detect_rknn.yaml

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/.."

if ! grep -q "ENABLE_RKNN:BOOL=ON" build/CMakeCache.txt 2>/dev/null; then
  echo "[SKIP] RKNN not enabled in build"
  exit 2
fi

if [ ! -x build/detect_cli ]; then
  echo "[INFO] Building project"
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_RKNN=ON
  cmake --build build -j
fi

CFG="config/detection/detect_rknn.yaml"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
OUT_LOG="$LOG_DIR/proof_rknn_$(date +%Y%m%d_%H%M%S).log"

if [ ! -f "$CFG" ]; then
  echo "[FAIL] Missing $CFG"
  exit 1
fi

echo "[RUN] detect_cli --cfg $CFG"
build/detect_cli --cfg "$CFG" 2>&1 | tee "$OUT_LOG" | cat || true

FRAMES=$(grep -E "^Frame [0-9]+: .*\([0-9]+ms\)" "$OUT_LOG" | wc -l || true)
AVG_MS=$(grep -Eo "\(([0-9]+)ms\)" "$OUT_LOG" | grep -Eo "[0-9]+" | awk '{sum+=$1; n++} END{ if(n>0) printf "%.1f", sum/n; else print 0 }')
if [ "${AVG_MS:-0}" != "0" ]; then
  FPS=$(awk -v ms="$AVG_MS" 'BEGIN{ if (ms>0) printf "%.1f", 1000.0/ms; else print 0 }')
else
  FPS=0
fi
echo "Frames: $FRAMES, Avg latency: ${AVG_MS} ms, Approx FPS: ${FPS}"

awk -v fps="$FPS" 'BEGIN{ exit (fps>=24)?0:1 }' && {
  echo "[PASS] FPS >= 24"
  exit 0
} || {
  echo "[FAIL] FPS < 24"
  exit 1
}


