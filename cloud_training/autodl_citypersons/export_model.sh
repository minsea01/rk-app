#!/bin/bash
# 导出训练好的模型到 ONNX 格式

set -e

echo "导出模型到 ONNX..."

MODEL="outputs/yolov8n_citypersons/weights/best.pt"

if [ ! -f "$MODEL" ]; then
    echo "错误: 未找到训练好的模型: $MODEL"
    exit 1
fi

# 导出 ONNX
yolo export \
    model=$MODEL \
    format=onnx \
    opset=12 \
    simplify=True \
    imgsz=640

echo ""
echo "导出完成!"
echo "  PT:   $MODEL"
echo "  ONNX: ${MODEL%.pt}.onnx"
echo ""

# 显示模型信息
python3 << 'PYEOF'
import os

pt_file = "outputs/yolov8n_citypersons/weights/best.pt"
onnx_file = "outputs/yolov8n_citypersons/weights/best.onnx"

if os.path.exists(pt_file):
    size_mb = os.path.getsize(pt_file) / 1024 / 1024
    print(f"PyTorch模型: {size_mb:.1f} MB")

if os.path.exists(onnx_file):
    size_mb = os.path.getsize(onnx_file) / 1024 / 1024
    print(f"ONNX模型: {size_mb:.1f} MB")

    # 预估 RKNN INT8 大小 (约为 ONNX 的 40%)
    rknn_est = size_mb * 0.4
    print(f"RKNN INT8预估: {rknn_est:.1f} MB")

    if rknn_est < 5.0:
        print("✅ 满足毕设要求 (<5MB)")
    else:
        print("⚠️ 可能超过毕设要求 (>5MB)")
PYEOF
