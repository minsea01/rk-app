#!/bin/bash
# COCO Person 极限训练 - 目标90% mAP
# 使用完整COCO数据集 + 更长训练时间 + 更强数据增强
# AutoDL 4090 预计训练时间: 8-12小时

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /root/autodl-tmp/pedestrian_training

echo "========================================="
echo "YOLOv8n COCO Person 极限训练"
echo "目标: 90% mAP50 (Person类)"
echo "========================================="

# 使用已训练的80%模型作为起点
PRETRAINED=""
if [ -f "outputs/yolov8n_person_v3/weights/best.pt" ]; then
    PRETRAINED="outputs/yolov8n_person_v3/weights/best.pt"
    echo "✅ 使用80% mAP预训练模型"
elif [ -f "models/yolov8n.pt" ]; then
    PRETRAINED="models/yolov8n.pt"
    echo "使用原始YOLOv8n"
fi

# COCO数据集
DATA="/root/coco_full.yaml"

echo ""
echo "策略: 长时间训练 + 强数据增强 + 小学习率微调"
echo ""

# 阶段1: 如果是从80%模型开始，用小学习率微调
if [[ "$PRETRAINED" == *"person"* ]]; then
    echo "=== 阶段1: 小学习率微调 ==="

    "$SCRIPT_DIR/train_runner.sh" \
      --profile extreme_stage1 \
      --model "$PRETRAINED" \
      --data "$DATA" \
      --epochs 200 \
      --imgsz 640 \
      --batch 64 \
      --device 0 \
      --project outputs \
      --name yolov8n_person_90_stage1 \
      --workers 16 \
      --cache ram \
      --classes 0 \
      --no-export

    PRETRAINED="outputs/yolov8n_person_90_stage1/weights/best.pt"
fi

# 阶段2: 更强增强继续训练
echo ""
echo "=== 阶段2: 强增强训练 ==="

"$SCRIPT_DIR/train_runner.sh" \
  --profile extreme_stage2 \
  --model "$PRETRAINED" \
  --data "$DATA" \
  --epochs 300 \
  --imgsz 640 \
  --batch 64 \
  --device 0 \
  --project outputs \
  --name yolov8n_person_90_final \
  --workers 16 \
  --cache ram \
  --classes 0 \
  --no-export

echo ""
echo "========================================="
echo "极限训练完成!"
echo "最佳模型: outputs/yolov8n_person_90_final/weights/best.pt"
echo "========================================="

# 检查最终mAP
python3 << 'PYEOF'
import csv
import os

for name in ['yolov8n_person_90_final', 'yolov8n_person_90_stage1']:
    results_file = f"outputs/{name}/results.csv"
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                last = rows[-1]
                for key in last.keys():
                    if 'mAP50' in key and 'mAP50-95' not in key:
                        val = float(last[key].strip())
                        print(f"{name}: mAP50 = {val*100:.1f}%")
                        if val >= 0.90:
                            print("  ✅ 达到90%目标!")
                        break
        break
PYEOF
