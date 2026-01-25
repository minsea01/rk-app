#!/bin/bash
# YOLOv8n 行人检测训练脚本
# AutoDL 4090 预计训练时间: 2-4小时 (100 epochs)

set -e

cd ~/pedestrian_training

echo "========================================="
echo "YOLOv8n 行人检测训练"
echo "========================================="

# 训练参数
MODEL="models/yolov8n.pt"
DATA="datasets/citypersons.yaml"  # 或 "coco8.yaml" 测试
EPOCHS=100
IMGSZ=640
BATCH=32  # 4090 24GB可用32-64
DEVICE=0
PROJECT="outputs"
NAME="yolov8n_pedestrian"

# 检查数据集
if [ ! -f "$DATA" ]; then
    echo "警告: $DATA 不存在"
    echo "使用COCO person子集进行快速测试..."
    DATA="coco8.yaml"
fi

# 开始训练
echo "开始训练..."
echo "  模型: $MODEL"
echo "  数据: $DATA"
echo "  Epochs: $EPOCHS"
echo "  Batch: $BATCH"
echo "  图片尺寸: $IMGSZ"
echo ""

yolo detect train \
    model=$MODEL \
    data=$DATA \
    epochs=$EPOCHS \
    imgsz=$IMGSZ \
    batch=$BATCH \
    device=$DEVICE \
    project=$PROJECT \
    name=$NAME \
    patience=20 \
    save=True \
    save_period=10 \
    val=True \
    plots=True \
    exist_ok=True \
    pretrained=True \
    optimizer=AdamW \
    lr0=0.001 \
    lrf=0.01 \
    momentum=0.937 \
    weight_decay=0.0005 \
    warmup_epochs=3 \
    warmup_momentum=0.8 \
    box=7.5 \
    cls=0.5 \
    dfl=1.5 \
    hsv_h=0.015 \
    hsv_s=0.7 \
    hsv_v=0.4 \
    degrees=0.0 \
    translate=0.1 \
    scale=0.5 \
    mosaic=1.0 \
    mixup=0.1 \
    workers=8

echo ""
echo "========================================="
echo "训练完成!"
echo "最佳模型: $PROJECT/$NAME/weights/best.pt"
echo "最后模型: $PROJECT/$NAME/weights/last.pt"
echo "========================================="
