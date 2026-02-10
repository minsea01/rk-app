#!/bin/bash
# 优化版 90% mAP 行人检测训练脚本
# 训练逻辑已收敛到 train_runner.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_RUNNER="$SCRIPT_DIR/train_runner.sh"

echo "========================================="
echo "YOLOv8n 行人检测 - 目标90% mAP"
echo "优化版: 解决RAM溢出问题"
echo "========================================="

# 工作目录
WORK_DIR="${WORK_DIR:-/root/autodl-tmp/pedestrian_training}"
cd "$WORK_DIR" || { echo "工作目录不存在: $WORK_DIR"; exit 1; }

# ===== 1. 检测可用数据集 =====
echo ""
echo "[1/4] 检测数据集..."

DATA=""
DATASET_NAME=""
if [ -f "datasets/merged/merged.yaml" ]; then
    DATA="datasets/merged/merged.yaml"
    DATASET_NAME="COCO+CrowdHuman联合"
elif [ -f "datasets/crowdhuman/crowdhuman.yaml" ]; then
    DATA="datasets/crowdhuman/crowdhuman.yaml"
    DATASET_NAME="CrowdHuman"
elif [ -f "datasets/coco_person.yaml" ]; then
    DATA="datasets/coco_person.yaml"
    DATASET_NAME="COCO Person"
elif [ -f "/root/coco_full.yaml" ]; then
    DATA="/root/coco_full.yaml"
    DATASET_NAME="COCO Full"
else
    echo "❌ 未找到数据集配置文件"
    echo "请先运行数据集准备脚本"
    exit 1
fi
echo "  使用数据集: $DATASET_NAME ($DATA)"

# ===== 2. 检测预训练模型 =====
echo ""
echo "[2/4] 检测预训练模型..."

PRETRAINED=""
if [ -f "outputs/yolov8n_person_80/weights/best.pt" ]; then
    PRETRAINED="outputs/yolov8n_person_80/weights/best.pt"
    echo "  ✅ 使用80% mAP预训练模型"
elif [ -f "outputs/yolov8n_pedestrian/weights/best.pt" ]; then
    PRETRAINED="outputs/yolov8n_pedestrian/weights/best.pt"
    echo "  ✅ 使用之前训练的模型"
elif [ -f "models/yolov8n.pt" ]; then
    PRETRAINED="models/yolov8n.pt"
    echo "  使用原始YOLOv8n"
else
    PRETRAINED="yolov8n.pt"
    echo "  将从网络下载YOLOv8n"
fi

# ===== 3. 训练参数配置 =====
echo ""
echo "[3/4] 配置训练参数..."

EPOCHS=${EPOCHS:-300}
IMGSZ=${IMGSZ:-640}
BATCH=${BATCH:-160}
DEVICE=${DEVICE:-0}
PROJECT="outputs"
NAME="yolov8n_pedestrian_90"
CACHE_MODE="${CACHE_MODE:-disk}"
WORKERS=${WORKERS:-10}

echo "  Epochs: $EPOCHS"
echo "  Batch: $BATCH"
echo "  Cache: $CACHE_MODE"
echo "  Workers: $WORKERS"
echo ""

# ===== 4. 开始训练 =====
echo "[4/4] 开始训练..."
echo ""
echo "系统资源:"
nvidia-smi --query-gpu=memory.total,memory.free --format=csv 2>/dev/null || true
echo "RAM: $(free -h | grep Mem | awk '{print $2}') 总量, $(free -h | grep Mem | awk '{print $7}') 可用"
echo ""

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

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
  --patience 80 \
  --save-period 25 \
  --workers "$WORKERS" \
  --cache "$CACHE_MODE" \
  --classes 0 \
  --no-export

echo ""
echo "========================================="
echo "训练完成!"
echo "最佳模型: $PROJECT/$NAME/weights/best.pt"
echo "========================================="
