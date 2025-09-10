#!/bin/bash
set -euo pipefail

# Non-interactive, evidence-producing demo script (实干版)
# - Mirrors the presentational demo steps, but performs real checks/actions
# - Builds with RKNN/GIGE if available
# - Runs detect_cli to report FPS
# - Optionally runs iperf3 for dual NIC throughput if endpoints are provided

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/.."

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
OUT_LOG="$LOG_DIR/proof_detect_$(date +%Y%m%d_%H%M%S).log"

echo "[1/8] 项目结构与统计 (真实扫描)"
echo "项目根目录: $(pwd)"
echo "主要目录:"
ls -la | grep "^d" | awk '{printf "%-20s %s\n", $9, "# 目录"}' | cat
echo "关键文件统计:"
printf "  源文件数量: %s\n" "$(find src -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.c" -o -name "*.h" \) 2>/dev/null | wc -l)"
printf "  配置文件数量: %s\n" "$(find config -type f \( -name "*.yaml" -o -name "*.yml" -o -name "*.txt" \) 2>/dev/null | wc -l)"
printf "  脚本数量: %s\n" "$(find scripts -type f \( -name "*.sh" -o -name "*.py" \) 2>/dev/null | wc -l)"
printf "  文档数量: %s\n" "$(find docs -type f -name "*.md" 2>/dev/null | wc -l)"

echo "[2/8] 模型与类别 (真实存在性与大小)"
mkdir -p artifacts/models
if [ ! -f artifacts/models/best.onnx ] && [ -f docs/models/best.onnx ]; then
  cp -f docs/models/best.onnx artifacts/models/best.onnx || true
fi
if [ -d artifacts/models ]; then
  ls -lh artifacts/models/ 2>/dev/null || true
  for m in artifacts/models/*.{pt,onnx,rknn}; do
    [ -f "$m" ] || continue
    echo "  $(basename "$m"): $(du -h "$m" | cut -f1)"
  done
fi
if [ -f config/industrial_classes.txt ]; then
  echo "工业类别总数: $(wc -l < config/industrial_classes.txt)"
else
  echo "[注意] 未找到 config/industrial_classes.txt"
fi

echo "[3/8] 配置文件检查 (真实文件内容摘要)"
if [ -f config/detection/detect.yaml ]; then
  echo "— config/detection/detect.yaml (前30行) —"; head -30 config/detection/detect.yaml | cat
else
  echo "[注意] 缺少 config/detection/detect.yaml"
fi
if [ -f config/deploy/rk3588_industrial_final.yaml ]; then
  echo "— config/deploy/rk3588_industrial_final.yaml (前25行) —"; head -25 config/deploy/rk3588_industrial_final.yaml | cat
else
  echo "[注意] 缺少 config/deploy/rk3588_industrial_final.yaml"
fi

echo "[4/8] 核心源码片段 (真实 grep)"
[ -f src/capture/GigeSource.cpp ] && { echo "— GigeSource::open —"; grep -n "bool GigeSource::open" -n src/capture/GigeSource.cpp -n | cat; }
[ -f src/infer/onnx/OnnxEngine.cpp ] && { echo "— OnnxEngine::init —"; grep -n "bool OnnxEngine::init" src/infer/onnx/OnnxEngine.cpp | head -1 | cat; }
[ -f examples/detect_cli.cpp ] && { echo "— detect_cli main/sources —"; grep -nE "(int main\(|source|engine)" examples/detect_cli.cpp | head -10 | cat; }

echo "[5/8] 配置与构建 (真实构建)"
BUILD_DIR="build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DENABLE_ONNX=ON
)

# Enable RKNN when SDK is present
if [ -d "/opt/rknpu2" ]; then
  CMAKE_ARGS+=( -DENABLE_RKNN=ON )
fi

# Enable GIGE if gstreamer and aravis are installed
if pkg-config --exists gstreamer-1.0 gstreamer-app-1.0; then
  CMAKE_ARGS+=( -DENABLE_GIGE=ON )
fi

cmake "${CMAKE_ARGS[@]}" .. | cat
cmake --build . -j | cat

echo "[6/8] 运行检测 (RKNN 优先，真实测量)"
cd ..
CFG_RKNN="config/detection/detect_rknn.yaml"
CFG_ONNX="config/detection/detect_demo.yaml"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
OUT_LOG="$LOG_DIR/proof_detect_$(date +%Y%m%d_%H%M%S).log"

# Detect RKNN enablement from CMakeCache
RKNN_ENABLED=0
if grep -q "ENABLE_RKNN:BOOL=ON" build/CMakeCache.txt 2>/dev/null; then
  RKNN_ENABLED=1
fi

# Ensure ONNX model exists (copy from docs if necessary)
if [ ! -f artifacts/models/best.onnx ]; then
  mkdir -p artifacts/models
  if [ -f docs/models/best.onnx ]; then
    cp -f docs/models/best.onnx artifacts/models/best.onnx || true
  fi
fi

# Ensure a small synthetic dataset exists for folder source
SYNTH_DIR="datasets/synth"
if [ ! -d "$SYNTH_DIR" ] || [ -z "$(ls -A "$SYNTH_DIR" 2>/dev/null)" ]; then
  echo "Generating synthetic dataset in $SYNTH_DIR"
  mkdir -p "$SYNTH_DIR"
  for i in $(seq 0 19); do
    F="$SYNTH_DIR/img_$i.pgm"
    W=640; H=480
    {
      echo "P2"; echo "$W $H"; echo "255";
      awk -v w=$W -v h=$H -v s=$i 'BEGIN{for(y=0;y<h;y++){for(x=0;x<w;x++){val=((x+y+s)%256); printf "%d ", val} printf "\n"}}'
    } > "$F"
  done
fi

if [ -f "$BUILD_DIR/detect_cli" ]; then
  if [ $RKNN_ENABLED -eq 1 ] && [ -f "$CFG_RKNN" ]; then
    echo "Running detect_cli with RKNN config: $CFG_RKNN"
    "$BUILD_DIR/detect_cli" --cfg "$CFG_RKNN" 2>&1 | tee "$OUT_LOG" | cat || true
  else
    echo "Running detect_cli with ONNX config: $CFG_ONNX (source override: $SYNTH_DIR)"
    "$BUILD_DIR/detect_cli" --cfg "$CFG_ONNX" --source "$SYNTH_DIR" 2>&1 | tee "$OUT_LOG" | cat || true
  fi
else
  echo "[ERROR] detect_cli not built"
  exit 2
fi

echo "[7/8] 性能汇总 (FPS 统计)"
FRAMES=$(grep -E "^Frame [0-9]+: .*\([0-9]+ms\)" "$OUT_LOG" | wc -l || true)
AVG_MS=$(grep -Eo "\(([0-9]+)ms\)" "$OUT_LOG" | grep -Eo "[0-9]+" | awk '{sum+=$1; n++} END{ if(n>0) printf "%.1f", sum/n; else print 0 }')
if [ "${AVG_MS:-0}" != "0" ]; then
  FPS=$(awk -v ms="$AVG_MS" 'BEGIN{ if (ms>0) printf "%.1f", 1000.0/ms; else print 0 }')
else
  FPS=0
fi
echo "Frames: $FRAMES, Avg latency: ${AVG_MS} ms, Approx FPS: ${FPS}"

PASS_FPS=0
awk -v fps="$FPS" 'BEGIN{ exit (fps>=24)?0:1 }' && PASS_FPS=1 || PASS_FPS=0

echo "[8/8] 双口吞吐 (可选 iperf3 实测)"
PASS_NET=2
if command -v iperf3 >/dev/null 2>&1 && [ -n "${IPERF_ETH0:-}" ] && [ -n "${IPERF_ETH1:-}" ]; then
  echo "Testing iperf3 on eth0 -> $IPERF_ETH0 and eth1 -> $IPERF_ETH1"
  iperf3 -c "$IPERF_ETH0" -t 5 -i 1 2>&1 | tee "$LOG_DIR/iperf_eth0.log" | cat
  iperf3 -c "$IPERF_ETH1" -t 5 -i 1 2>&1 | tee "$LOG_DIR/iperf_eth1.log" | cat
  BW0=$(grep -E "sender|receiver" "$LOG_DIR/iperf_eth0.log" | tail -1 | awk '{print $(NF-1)}')
  BW1=$(grep -E "sender|receiver" "$LOG_DIR/iperf_eth1.log" | tail -1 | awk '{print $(NF-1)}')
  echo "eth0: ${BW0} Mbits/sec, eth1: ${BW1} Mbits/sec"
  awk -v b0="$BW0" -v b1="$BW1" 'BEGIN{ exit (b0>=900 && b1>=900)?0:1 }' && PASS_NET=1 || PASS_NET=0
else
  echo "iperf3 or endpoints not configured; skip network test (set IPERF_ETH0/IPERF_ETH1)"
fi

echo "[5/5] Verdict"
if [ $PASS_FPS -eq 1 ]; then echo "FPS: PASS (>=24)"; else echo "FPS: FAIL (<24)"; fi
if [ $PASS_NET -eq 1 ]; then echo "NET: PASS (>=900 Mbps each)"; elif [ $PASS_NET -eq 0 ]; then echo "NET: FAIL"; else echo "NET: SKIPPED"; fi

if [ $PASS_FPS -eq 1 ] && [ $PASS_NET != 0 ]; then exit 0; else exit 1; fi


