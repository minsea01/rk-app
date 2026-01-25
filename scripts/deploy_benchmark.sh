#!/bin/bash
# 部署benchmark测试文件到RK3588板端

BOARD_IP="192.168.137.226"
BOARD_USER="root"
BOARD_DIR="~/rk-app"

echo "================================"
echo "部署Benchmark测试文件到RK3588"
echo "================================"

echo ""
echo "1. 传输benchmark脚本..."
scp apps/benchmark_e2e_latency.py ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/apps/

echo ""
echo "2. 传输测试图像..."
scp assets/bus.jpg ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/assets/
scp assets/test.jpg ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/assets/

echo ""
echo "3. 传输RKNN模型（如需要）..."
# scp artifacts/models/yolo11n_416.rknn ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/artifacts/models/

echo ""
echo "================================"
echo "部署完成！"
echo "================================"
echo ""
echo "接下来请SSH到板端执行测试："
echo "  ssh ${BOARD_USER}@${BOARD_IP}"
echo ""
echo "板端测试命令："
echo "  cd ~/rk-app"
echo "  python3 apps/benchmark_e2e_latency.py \\"
echo "    --model artifacts/models/yolo11n_416.rknn \\"
echo "    --image assets/bus.jpg \\"
echo "    --imgsz 416 \\"
echo "    --simulate-1080p \\"
echo "    --runs 50 \\"
echo "    --output artifacts/e2e_latency_report.json"
echo ""
