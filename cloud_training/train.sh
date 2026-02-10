#!/bin/bash
# YOLOv8n 行人检测训练脚本（基础版）
# 训练逻辑已收敛到 train_runner.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_RUNNER="$SCRIPT_DIR/train_runner.sh"

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

echo "开始训练..."
echo "  模型: $MODEL"
echo "  数据: $DATA"
echo "  Epochs: $EPOCHS"
echo "  Batch: $BATCH"
echo "  图片尺寸: $IMGSZ"
echo ""

"$TRAIN_RUNNER" \
  --profile baseline \
  --model "$MODEL" \
  --data "$DATA" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --project "$PROJECT" \
  --name "$NAME" \
  --patience 20 \
  --save-period 10 \
  --optimizer AdamW \
  --lr0 0.001 \
  --lrf 0.01 \
  --momentum 0.937 \
  --weight-decay 0.0005 \
  --warmup-epochs 3 \
  --warmup-momentum 0.8 \
  --workers 8 \
  --extra "box=7.5" \
  --extra "cls=0.5" \
  --extra "dfl=1.5" \
  --extra "hsv_h=0.015" \
  --extra "hsv_s=0.7" \
  --extra "hsv_v=0.4" \
  --extra "degrees=0.0" \
  --extra "mixup=0.1" \
  --no-export

echo ""
echo "========================================="
echo "训练完成!"
echo "最佳模型: $PROJECT/$NAME/weights/best.pt"
echo "最后模型: $PROJECT/$NAME/weights/last.pt"
echo "========================================="
