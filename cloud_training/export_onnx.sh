#!/bin/bash
# 训练完成后导出ONNX模型

set -e

cd ~/pedestrian_training

BEST_PT="outputs/yolov8n_pedestrian/weights/best.pt"

if [ ! -f "$BEST_PT" ]; then
    echo "错误: $BEST_PT 不存在"
    echo "请先运行训练脚本"
    exit 1
fi

echo "========================================="
echo "导出ONNX模型"
echo "========================================="

# 导出640x640 ONNX (标准)
echo "[1/2] 导出 640x640 ONNX..."
yolo export model=$BEST_PT format=onnx imgsz=640 simplify=True opset=12

# 导出416x416 ONNX (NPU优化)
echo "[2/2] 导出 416x416 ONNX (NPU优化)..."
yolo export model=$BEST_PT format=onnx imgsz=416 simplify=True opset=12

echo ""
echo "========================================="
echo "导出完成!"
echo "文件位置:"
ls -lh outputs/yolov8n_pedestrian/weights/*.onnx 2>/dev/null || echo "  (检查 outputs/ 目录)"
echo ""
echo "下载这些文件回本地进行RKNN转换:"
echo "  scp root@<autodl_ip>:~/pedestrian_training/outputs/yolov8n_pedestrian/weights/best*.onnx ."
echo "========================================="
