#!/bin/bash
set -euo pipefail

# 行人检测模型训练脚本
# 用法: bash scripts/train_person_detector.sh [yolo11s|yolo11m|yolo11n]

MODEL=${1:-yolo11s}  # 默认使用 yolo11s (更大的模型)
IMGSZ=416
EPOCHS=100
BATCH=16

echo "=== COCO 行人检测模型训练 ==="
echo "模型: $MODEL"
echo "分辨率: $IMGSZ"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH"
echo ""

# 激活环境
source ~/yolo_env/bin/activate
cd /home/minsea/rk-app

# 第一步: 准备数据集
echo "第一步: 准备 COCO 行人子集..."
if [ ! -d "datasets/coco_person" ]; then
    python3 scripts/prepare_person_dataset.py
else
    echo "数据集已存在，跳过准备步骤"
fi

# 第二步: 训练模型
echo ""
echo "第二步: 开始训练 $MODEL 模型..."
yolo detect train \
    model=$MODEL.pt \
    data=datasets/coco_person/data.yaml \
    epochs=$EPOCHS \
    imgsz=$IMGSZ \
    batch=$BATCH \
    patience=20 \
    save=True \
    project=runs/detect \
    name=person_${MODEL}_${IMGSZ} \
    exist_ok=False \
    pretrained=True \
    optimizer=auto \
    verbose=True \
    device=0

# 第三步: 验证模型
echo ""
echo "第三步: 验证训练好的模型..."
BEST_MODEL="runs/detect/person_${MODEL}_${IMGSZ}/weights/best.pt"

yolo detect val \
    model=$BEST_MODEL \
    data=datasets/coco_person/data.yaml \
    imgsz=$IMGSZ \
    batch=$BATCH \
    device=0

# 第四步: 导出 ONNX
echo ""
echo "第四步: 导出 ONNX 模型..."
yolo export \
    model=$BEST_MODEL \
    format=onnx \
    imgsz=$IMGSZ \
    simplify=True \
    opset=12

ONNX_MODEL="runs/detect/person_${MODEL}_${IMGSZ}/weights/best.onnx"

# 第五步: 转换 RKNN
echo ""
echo "第五步: 转换 ONNX 到 RKNN..."
python3 tools/convert_onnx_to_rknn.py \
    --onnx "$ONNX_MODEL" \
    --out "artifacts/models/person_${MODEL}_${IMGSZ}.rknn" \
    --calib datasets/coco/calib_images/calib.txt \
    --target rk3588 \
    --do-quant

echo ""
echo "=== 训练完成! ==="
echo "PyTorch 模型: $BEST_MODEL"
echo "ONNX 模型: $ONNX_MODEL"
echo "RKNN 模型: artifacts/models/person_${MODEL}_${IMGSZ}.rknn"
echo ""
echo "下一步:"
echo "1. 检查 runs/detect/person_${MODEL}_${IMGSZ}/results.csv 查看训练曲线"
echo "2. 用 PC 测试 ONNX 性能"
echo "3. 用 RKNN 模拟器测试精度"
echo "4. 更新 artifacts/models/best.* 链接到新模型"

