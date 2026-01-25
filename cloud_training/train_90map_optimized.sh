#!/bin/bash
# 优化版 90% mAP 行人检测训练脚本
# 解决 RAM 溢出问题：禁用 cache=ram，使用磁盘缓存
# AutoDL 4090 预计训练时间: 6-10小时
# 支持: COCO Person + CrowdHuman 联合训练

set -e

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

# 优先级: 联合数据集 > CrowdHuman > COCO Person > COCO Full
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

# 核心参数
EPOCHS=${EPOCHS:-300}
IMGSZ=${IMGSZ:-640}
BATCH=${BATCH:-160}       # RTX 5090 32GB: batch=160-192 可用
DEVICE=${DEVICE:-0}
PROJECT="outputs"
NAME="yolov8n_pedestrian_90"

# 缓存策略
# ⚠️ 重要: COCO+CrowdHuman (79k图) cache=ram 需要 ~108GB
#          90GB RAM 不够用 cache=ram!
# - disk: 安全，速度稍慢 (推荐)
# - False: 不缓存，最慢但最安全
# - ram: 仅适用于单独 COCO Person (64k, ~77GB)
CACHE_MODE="${CACHE_MODE:-disk}"

# workers数量 (25 vCPU Xeon Platinum)
# 减少workers可降低内存峰值
WORKERS=${WORKERS:-10}

echo "  Epochs: $EPOCHS"
echo "  Batch: $BATCH"
echo "  Cache: $CACHE_MODE"
echo "  Workers: $WORKERS"
echo ""

# ===== 4. 开始训练 =====
echo "[4/4] 开始训练..."
echo ""

# 显示系统资源
echo "系统资源:"
nvidia-smi --query-gpu=memory.total,memory.free --format=csv 2>/dev/null || true
echo "RAM: $(free -h | grep Mem | awk '{print $2}') 总量, $(free -h | grep Mem | awk '{print $7}') 可用"
echo ""

# 设置环境变量优化内存使用
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 训练命令
yolo detect train \
    model=$PRETRAINED \
    data=$DATA \
    epochs=$EPOCHS \
    imgsz=$IMGSZ \
    batch=$BATCH \
    device=$DEVICE \
    project=$PROJECT \
    name=$NAME \
    patience=80 \
    save=True \
    save_period=25 \
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
    degrees=10.0 \
    translate=0.15 \
    scale=0.6 \
    shear=5.0 \
    perspective=0.0005 \
    flipud=0.0 \
    fliplr=0.5 \
    mosaic=1.0 \
    mixup=0.15 \
    copy_paste=0.1 \
    erasing=0.4 \
    workers=$WORKERS \
    cache=$CACHE_MODE \
    amp=True \
    classes=0

echo ""
echo "========================================="
echo "训练完成!"
echo "最佳模型: $PROJECT/$NAME/weights/best.pt"
echo "========================================="

# 检查最终mAP
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
            for key in last.keys():
                if 'mAP50' in key and 'mAP50-95' not in key:
                    try:
                        val = float(last[key].strip())
                        print(f"\n最终 mAP50: {val*100:.1f}%")
                        if val >= 0.90:
                            print("✅ 达到90% mAP目标!")
                        elif val >= 0.85:
                            print(f"⚠️  接近目标，距离90%还差 {(0.90-val)*100:.1f}%")
                            print("   建议: 继续训练或使用更多数据")
                        else:
                            print(f"⚠️  距离90%还差 {(0.90-val)*100:.1f}%")
                            print("   建议: 1) 合并CrowdHuman 2) 增加epochs 3) 调整超参")
                    except:
                        pass
                    break
PYEOF
