#!/bin/bash
# 90% mAP 高精度行人检测训练脚本
# 训练逻辑已收敛到 train_runner.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_RUNNER="$SCRIPT_DIR/train_runner.sh"

cd ~/pedestrian_training

echo "========================================="
echo "YOLOv8n 行人检测 - 目标90% mAP"
echo "========================================="

# 检查是否有预训练的80%模型
PRETRAINED=""
if [ -f "outputs/yolov8n_person_v3/weights/best.pt" ]; then
    PRETRAINED="outputs/yolov8n_person_v3/weights/best.pt"
    echo "发现预训练模型: $PRETRAINED"
elif [ -f "models/yolov8n_person_80map.pt" ]; then
    PRETRAINED="models/yolov8n_person_80map.pt"
    echo "发现预训练模型: $PRETRAINED"
else
    PRETRAINED="models/yolov8n.pt"
    echo "使用原始YOLOv8n: $PRETRAINED"
fi

# 选择数据集
DATA=""
if [ -f "datasets/widerperson/widerperson.yaml" ]; then
    DATA="datasets/widerperson/widerperson.yaml"
elif [ -f "datasets/citypersons.yaml" ]; then
    DATA="datasets/citypersons.yaml"
else
    DATA="/root/coco_full.yaml"
fi

echo "使用数据集: $DATA"

# 训练参数
EPOCHS=300
IMGSZ=640
BATCH=64
DEVICE=0
PROJECT="outputs"
NAME="yolov8n_pedestrian_90"

echo ""
echo "训练配置:"
echo "  模型: $PRETRAINED"
echo "  数据: $DATA"
echo "  Epochs: $EPOCHS"
echo "  Batch: $BATCH"
echo "  目标: 90% mAP50"
echo ""

"$TRAIN_RUNNER" \
  --profile map90 \
  --model "$PRETRAINED" \
  --data "$DATA" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --project "$PROJECT" \
  --name "$NAME" \
  --patience 100 \
  --save-period 20 \
  --workers 16 \
  --cache ram \
  --classes 0 \
  --extra "degrees=5.0" \
  --extra "shear=2.0" \
  --extra "crop_fraction=0.1" \
  --no-export

echo ""
echo "========================================="
echo "训练完成!"
echo "最佳模型: $PROJECT/$NAME/weights/best.pt"
echo ""
echo "查看结果:"
echo "  mAP: grep 'mAP50' $PROJECT/$NAME/results.csv | tail -1"
echo "========================================="
