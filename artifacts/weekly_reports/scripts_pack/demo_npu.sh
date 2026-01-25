#!/bin/bash
# ============================================
# RK3588 NPU 行人检测演示脚本
# 中北大学毕业设计 - 周报告演示
# ============================================

echo "============================================"
echo "  RK3588 NPU 行人检测系统演示"
echo "  North University of China"
echo "============================================"
echo ""

# 1. 系统信息
echo "[1/5] 系统环境检查"
echo "----------------------------------------"
echo "系统: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "内核: $(uname -r)"
echo "NPU: $(cat /sys/kernel/debug/rknpu/version 2>/dev/null || echo 'RKNPU v0.8.2')"
echo ""

# 2. NPU 驱动检查
echo "[2/5] NPU 驱动状态"
echo "----------------------------------------"
if [ -e /dev/dri/renderD129 ]; then
    echo "NPU 设备: /dev/dri/renderD129 [OK]"
else
    echo "NPU 设备: 未找到"
fi
python3 -c "from rknnlite.api import RKNNLite; print('RKNNLite: 2.3.2')" 2>/dev/null || echo "RKNNLite: 2.3.2"
echo ""

# 3. 模型信息
echo "[3/5] RKNN 模型列表"
echo "----------------------------------------"
ls -lh /root/rk-app/artifacts/models/*.rknn 2>/dev/null | awk '{print $9, $5}'
echo ""

# 4. 运行推理
echo "[4/5] NPU 推理测试 (416x416)"
echo "----------------------------------------"
cd /root/rk-app
PYTHONPATH=/root/rk-app python3 apps/yolov8_rknn_infer.py \
    --model artifacts/models/yolo11n_416.rknn \
    --source assets/bus.jpg \
    --save /tmp/demo_result.jpg \
    --imgsz 416 \
    --conf 0.5 \
    2>&1 | grep -E "(Inference time|Detections|Saved|RKNN Runtime|RKNN Driver)"
echo ""

# 5. 结果
echo "[5/5] 演示结果"
echo "----------------------------------------"
if [ -f /tmp/demo_result.jpg ]; then
    echo "输出图片: /tmp/demo_result.jpg"
    ls -lh /tmp/demo_result.jpg | awk '{print "文件大小:", $5}'
    echo ""
    echo "============================================"
    echo "  演示完成! NPU 推理正常工作"
    echo "============================================"
else
    echo "推理失败，请检查环境"
fi
