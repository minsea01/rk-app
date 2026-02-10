#!/bin/bash
# YOLOv8n CityPersons/WiderPerson 训练脚本
# 目标: mAP@0.5 >= 90%

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_RUNNER="$SCRIPT_DIR/../train_runner.sh"

echo "============================================"
echo "  YOLOv8n 行人检测训练"
echo "============================================"

# 配置
EPOCHS=${EPOCHS:-150}
BATCH=${BATCH:-64}        # RTX 4090: 64-128
IMGSZ=${IMGSZ:-640}
WORKERS=${WORKERS:-8}

# 数据集配置
if [ -f "datasets/person.yaml" ]; then
    DATA="datasets/person.yaml"
elif [ -f "datasets/widerperson/widerperson.yaml" ]; then
    DATA="datasets/widerperson/widerperson.yaml"
else
    echo "错误: 未找到数据集配置文件"
    exit 1
fi

echo "数据集: $DATA"
echo "Epochs: $EPOCHS"
echo "Batch: $BATCH"
echo ""

# 检查预训练模型
MODEL="yolov8n.pt"
if [ -f "outputs/yolov8n_citypersons/weights/last.pt" ]; then
    echo "发现之前的训练，从断点继续..."
    MODEL="outputs/yolov8n_citypersons/weights/last.pt"
fi

"$TRAIN_RUNNER" \
  --profile baseline \
  --model "$MODEL" \
  --data "$DATA" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --device 0 \
  --project outputs \
  --name yolov8n_citypersons \
  --patience 50 \
  --save-period 10 \
  --workers "$WORKERS" \
  --cache disk \
  --optimizer AdamW \
  --lr0 0.001 \
  --lrf 0.01 \
  --momentum 0.937 \
  --weight-decay 0.0005 \
  --warmup-epochs 3 \
  --extra "box=7.5" \
  --extra "cls=0.5" \
  --extra "dfl=1.5" \
  --extra "hsv_h=0.015" \
  --extra "hsv_s=0.7" \
  --extra "hsv_v=0.4" \
  --extra "shear=2.0" \
  --extra "mixup=0.1" \
  --no-export

echo ""
echo "训练完成!"
