#!/bin/bash
# YOLOv8n CityPersons/WiderPerson 训练脚本
# 目标: mAP@0.5 >= 90%

set -e

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

# 开始训练
yolo detect train \
    model=$MODEL \
    data=$DATA \
    epochs=$EPOCHS \
    imgsz=$IMGSZ \
    batch=$BATCH \
    device=0 \
    project=outputs \
    name=yolov8n_citypersons \
    patience=50 \
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
    box=7.5 \
    cls=0.5 \
    dfl=1.5 \
    hsv_h=0.015 \
    hsv_s=0.7 \
    hsv_v=0.4 \
    degrees=5.0 \
    translate=0.1 \
    scale=0.5 \
    shear=2.0 \
    flipud=0.0 \
    fliplr=0.5 \
    mosaic=1.0 \
    mixup=0.1 \
    copy_paste=0.1 \
    workers=$WORKERS \
    cache=disk \
    amp=True

echo ""
echo "训练完成!"

# 显示结果
python3 << 'PYEOF'
import csv
import os

results_file = "outputs/yolov8n_citypersons/results.csv"
if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if rows:
            last = rows[-1]
            for key in last.keys():
                if 'mAP50' in key and 'mAP50-95' not in key:
                    try:
                        val = float(last[key].strip())
                        print(f"\n最终 mAP@0.5: {val*100:.1f}%")
                        if val >= 0.90:
                            print("✅ 达到 90% mAP 目标!")
                        elif val >= 0.85:
                            print(f"⚠️ 接近目标，差 {(0.90-val)*100:.1f}%")
                        else:
                            print(f"⚠️ 未达标，差 {(0.90-val)*100:.1f}%")
                    except:
                        pass
                    break
PYEOF
