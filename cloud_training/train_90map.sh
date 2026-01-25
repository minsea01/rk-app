#!/bin/bash
# 90% mAP 高精度行人检测训练脚本
# 使用预训练的80% COCO Person模型进行微调
# AutoDL 4090 预计训练时间: 4-8小时

set -e

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
    # 使用COCO Person
    DATA="/root/coco_full.yaml"
fi

echo "使用数据集: $DATA"

# 训练参数 - 针对90%+ mAP优化
EPOCHS=300
IMGSZ=640
BATCH=64          # 4090可以用64-128
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

# 高精度训练
yolo detect train \
    model=$PRETRAINED \
    data=$DATA \
    epochs=$EPOCHS \
    imgsz=$IMGSZ \
    batch=$BATCH \
    device=$DEVICE \
    project=$PROJECT \
    name=$NAME \
    patience=100 \
    save=True \
    save_period=20 \
    val=True \
    plots=True \
    exist_ok=True \
    pretrained=True \
    optimizer=AdamW \
    lr0=0.0005 \
    lrf=0.001 \
    momentum=0.937 \
    weight_decay=0.0005 \
    warmup_epochs=5 \
    warmup_momentum=0.8 \
    box=7.5 \
    cls=0.5 \
    dfl=1.5 \
    hsv_h=0.02 \
    hsv_s=0.8 \
    hsv_v=0.5 \
    degrees=5.0 \
    translate=0.15 \
    scale=0.6 \
    shear=2.0 \
    perspective=0.0005 \
    flipud=0.0 \
    fliplr=0.5 \
    mosaic=1.0 \
    mixup=0.15 \
    copy_paste=0.1 \
    erasing=0.4 \
    crop_fraction=0.1 \
    workers=16 \
    cache=ram \
    amp=True \
    classes=0

echo ""
echo "========================================="
echo "训练完成!"
echo "最佳模型: $PROJECT/$NAME/weights/best.pt"
echo ""
echo "查看结果:"
echo "  mAP: grep 'mAP50' $PROJECT/$NAME/results.csv | tail -1"
echo "========================================="

# 自动检查是否达到90%
python3 << 'PYEOF'
import csv
import os

results_file = "outputs/yolov8n_pedestrian_90/results.csv"
if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if rows:
            last = rows[-1]
            # 找到mAP50列
            for key in last.keys():
                if 'mAP50' in key and 'mAP50-95' not in key:
                    val = float(last[key].strip())
                    print(f"\n最终 mAP50: {val*100:.1f}%")
                    if val >= 0.90:
                        print("✅ 达到90% mAP目标!")
                    else:
                        print(f"⚠️  距离90%还差 {(0.90-val)*100:.1f}%")
                    break
PYEOF
