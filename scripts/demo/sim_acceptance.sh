#!/usr/bin/env bash
set -euo pipefail

# 纯软件仿真验收：
# 1) ONNXRuntime基准（avg/p50/p90/FPS）
# 2) 生成可视化预测图
# 3) 汇总到 artifacts/reports/sim_summary.md

ROOT=$(cd "$(dirname "$0")/.." && pwd)
ONNX=${1:-$ROOT/check3_fuse_ops.onnx}
IMG=${2:-}
IMGSZ=${3:-640}
LOOPS=${4:-200}

PY=python3

OUT_DIR=$ROOT/artifacts
REP_DIR=$OUT_DIR/reports
LOG_DIR=$OUT_DIR/logs
VIS_OUT=$OUT_DIR/vis/sim_out.jpg
mkdir -p "$OUT_DIR/vis" "$REP_DIR" "$LOG_DIR/fps"

echo "[SIM] Using ONNX: $ONNX"
if [ ! -f "$ONNX" ]; then
  echo "[ERR] ONNX not found: $ONNX" >&2
  exit 1
fi

echo "[SIM] Benchmarking with ONNXRuntime..."
BENCH_LOG=$LOG_DIR/fps/onnx_bench.txt
$PY "$ROOT/tools/onnx_bench.py" --onnx "$ONNX" ${IMG:+--img "$IMG"} --imgsz "$IMGSZ" --loops "$LOOPS" --warmup 20 | tee "$BENCH_LOG"

echo "[SIM] Visualizing inference..."
$PY "$ROOT/tools/visualize_inference.py" --onnx "$ONNX" ${IMG:+--img "$IMG"} --imgsz "$IMGSZ" --out "$VIS_OUT" | tee "$LOG_DIR/fps/vis.txt"

AVG_MS=$(awk -F'=' '/avg_ms/{print $2}' "$BENCH_LOG" | awk '{print $1}')
FPS=$(awk -F'=' '/fps_avg/{print $3}' "$BENCH_LOG" | awk '{print $1}')
P50=$(awk -F'=' '/p50_ms/{print $2}' "$BENCH_LOG" | awk '{print $1}')
P90=$(awk -F'=' '/p90_ms/{print $2}' "$BENCH_LOG" | awk '{print $1}')

SUMMARY=$REP_DIR/sim_summary.md
{
  echo "# 软件仿真验收报告"
  echo
  echo "## ONNXRuntime 性能"
  echo "- 模型: $(basename "$ONNX")"
  echo "- 输入: ${IMGSZ}x${IMGSZ}"
  echo "- 循环: $LOOPS (warmup 20)"
  echo "- 平均延迟: ${AVG_MS:-N/A} ms"
  echo "- P50: ${P50:-N/A} ms, P90: ${P90:-N/A} ms"
  echo "- 平均FPS: ${FPS:-N/A}"
  echo
  echo "## 可视化结果"
  if [ -f "$VIS_OUT" ]; then
    echo "- 结果图片: $VIS_OUT"
  else
    echo "- 结果图片: 生成失败"
  fi
  echo
  echo "## 结论"
  echo "- 纯软件环境用于结构/后处理验证与相对性能参考。"
  echo "- 板端目标（≥24 fps、双口≥900Mbps/口）需在实际硬件上验收。"
} > "$SUMMARY"

echo "[SIM] Done. Summary -> $SUMMARY"


