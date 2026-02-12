#!/bin/bash
# 一键测试DFL优化：交叉编译 + 传输 + 板端运行
#
# 使用方法（在WSL执行）：
#   cd ~/rk-app
#   bash scripts/test_dfl_optimization.sh

BOARD_IP="192.168.137.226"
BOARD_USER="root"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_BIN="${ROOT_DIR}/artifacts/bin/bench_dfl_opt"
REMOTE_BIN_DIR="$HOME/rk-app/artifacts/bin"

set -e

echo "=================================================="
echo "DFL优化一键测试流程"
echo "=================================================="

# 步骤1: 交叉编译
echo ""
echo "[1/3] 交叉编译性能测试程序..."
bash scripts/build_dfl_bench_cross.sh

# 步骤2: 传输到板端
echo ""
echo "[2/3] 传输到板端 (${BOARD_USER}@${BOARD_IP})..."
ssh ${BOARD_USER}@${BOARD_IP} "mkdir -p ${REMOTE_BIN_DIR}"
scp "${LOCAL_BIN}" "${BOARD_USER}@${BOARD_IP}:${REMOTE_BIN_DIR}/"

# 步骤3: 板端运行
echo ""
echo "[3/3] 板端执行测试..."
echo "=================================================="
ssh ${BOARD_USER}@${BOARD_IP} "cd ~/rk-app && ./artifacts/bin/bench_dfl_opt"

echo ""
echo "=================================================="
echo "测试完成！"
echo "=================================================="
