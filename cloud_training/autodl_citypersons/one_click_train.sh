#!/bin/bash
# ============================================================
#  AutoDL 一键训练脚本 - COCO Person 行人检测
#  目标: YOLOv8n mAP@0.5 >= 90%
# ============================================================
#  使用方法:
#    上传后执行: chmod +x one_click_train.sh && ./one_click_train.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_RUNNER="$SCRIPT_DIR/../train_runner.sh"

echo "============================================================"
echo "  YOLOv8n 行人检测训练 - 目标 90% mAP"
echo "  AutoDL 4090 预计: 2-4 小时"
echo "============================================================"
echo ""

# ==================== 1. 环境准备 ====================
echo "[1/6] 安装依赖..."
pip install ultralytics>=8.0.0 opencv-python-headless --quiet
pip install pycocotools --quiet

# ==================== 2. 工作目录 ====================
WORK_DIR="/root/autodl-tmp/person_detection"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
echo "  工作目录: $WORK_DIR"

# ==================== 3. 数据集准备 ====================
echo ""
echo "[2/6] 准备 COCO Person 数据集..."

# 检查 COCO 数据集位置 (AutoDL 常见位置)
COCO_ROOT=""
for path in "/root/autodl-tmp/coco" "/root/coco" "/data/coco" "$HOME/datasets/coco"; do
    if [ -d "$path/train2017" ]; then
        COCO_ROOT="$path"
        break
    fi
done

if [ -z "$COCO_ROOT" ]; then
    echo "  未找到 COCO 数据集，正在下载..."
    mkdir -p datasets/coco
    cd datasets/coco

    # 下载 COCO 2017 (如果没有)
    if [ ! -d "train2017" ]; then
        echo "  下载 train2017 (约 18GB)..."
        wget -q --show-progress http://images.cocodataset.org/zips/train2017.zip
        unzip -q train2017.zip && rm train2017.zip
    fi

    if [ ! -d "val2017" ]; then
        echo "  下载 val2017 (约 1GB)..."
        wget -q --show-progress http://images.cocodataset.org/zips/val2017.zip
        unzip -q val2017.zip && rm val2017.zip
    fi

    if [ ! -d "annotations" ]; then
        echo "  下载 annotations..."
        wget -q --show-progress http://images.cocodataset.org/annotations/annotations_trainval2017.zip
        unzip -q annotations_trainval2017.zip && rm annotations_trainval2017.zip
    fi

    cd "$WORK_DIR"
    COCO_ROOT="$WORK_DIR/datasets/coco"
fi

echo "  COCO 数据集: $COCO_ROOT"

# ==================== 4. 筛选 Person 类 ====================
echo ""
echo "[3/6] 筛选 Person 类标注..."

python3 << PYEOF
import json
import os
from pathlib import Path

coco_root = "$COCO_ROOT"
output_dir = Path("$WORK_DIR/datasets/coco_person")
output_dir.mkdir(parents=True, exist_ok=True)

def filter_person_annotations(split):
    ann_file = f"{coco_root}/annotations/instances_{split}2017.json"
    if not os.path.exists(ann_file):
        print(f"  警告: {ann_file} 不存在")
        return

    print(f"  处理 {split}2017...")
    with open(ann_file) as f:
        coco = json.load(f)

    # 找到 person 类别 ID
    person_cat_id = None
    for cat in coco['categories']:
        if cat['name'] == 'person':
            person_cat_id = cat['id']
            break

    if person_cat_id is None:
        print("  错误: 未找到 person 类别")
        return

    # 筛选 person 标注
    person_anns = [a for a in coco['annotations'] if a['category_id'] == person_cat_id]
    person_img_ids = set(a['image_id'] for a in person_anns)
    person_images = [i for i in coco['images'] if i['id'] in person_img_ids]

    # 创建 YOLO 格式标注
    labels_dir = output_dir / "labels" / split
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 按图片分组
    img_id_to_anns = {}
    for ann in person_anns:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)

    # 图片 ID 到尺寸映射
    img_id_to_info = {i['id']: i for i in coco['images']}

    count = 0
    for img_id, anns in img_id_to_anns.items():
        img_info = img_id_to_info[img_id]
        img_w, img_h = img_info['width'], img_info['height']
        file_name = img_info['file_name'].replace('.jpg', '.txt')

        lines = []
        for ann in anns:
            x, y, w, h = ann['bbox']  # COCO: [x, y, width, height]
            # 转 YOLO: [x_center, y_center, width, height] 归一化
            x_center = (x + w/2) / img_w
            y_center = (y + h/2) / img_h
            width = w / img_w
            height = h / img_h

            if 0 <= x_center <= 1 and 0 <= y_center <= 1 and width > 0 and height > 0:
                lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        if lines:
            with open(labels_dir / file_name, 'w') as f:
                f.writelines(lines)
            count += 1

    print(f"  {split}: {count} 张图片, {len(person_anns)} 个行人标注")

filter_person_annotations('train')
filter_person_annotations('val')

# 创建数据集配置文件
yaml_content = f'''# COCO Person Dataset
path: {coco_root}
train: train2017
val: val2017

# 使用自定义标注
labels_path: {output_dir}/labels

names:
  0: person

nc: 1
'''

with open(output_dir / "coco_person.yaml", 'w') as f:
    f.write(yaml_content)

# 创建标准 YOLO 格式配置 (使用软链接)
yaml_standard = f'''# COCO Person Dataset (YOLO format)
path: {output_dir}
train: ../../../{coco_root}/train2017
val: ../../../{coco_root}/val2017

names:
  0: person

nc: 1
'''

print("  数据集准备完成!")
PYEOF

# ==================== 5. 开始训练 ====================
echo ""
echo "[4/6] 开始训练..."
echo "  Epochs: 100"
echo "  Batch: 64 (RTX 4090)"
echo ""

# 检测 GPU
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "警告: 未检测到 GPU"

# 创建数据集配置
cat > "$WORK_DIR/person.yaml" << EOF
# COCO Person Dataset
path: $COCO_ROOT
train: train2017
val: val2017

names:
  0: person

nc: 1
EOF

"$TRAIN_RUNNER" \
  --profile baseline \
  --workdir "$WORK_DIR" \
  --model yolov8n.pt \
  --data "$WORK_DIR/person.yaml" \
  --epochs 100 \
  --imgsz 640 \
  --batch 64 \
  --device 0 \
  --project outputs \
  --name yolov8n_person \
  --patience 30 \
  --save-period 10 \
  --workers 8 \
  --cache disk \
  --optimizer AdamW \
  --lr0 0.001 \
  --lrf 0.01 \
  --extra "mixup=0.1" \
  --classes 0

echo ""
echo "============================================================"
echo "  训练完成!"
echo "============================================================"
echo ""
echo "模型位置:"
echo "  PT:   $WORK_DIR/outputs/yolov8n_person/weights/best.pt"
echo "  ONNX: $WORK_DIR/outputs/yolov8n_person/weights/best.onnx"
echo ""
echo "下载命令 (本地执行):"
echo "  scp -P <端口> root@<地址>:$WORK_DIR/outputs/yolov8n_person/weights/best.pt ./"
echo "  scp -P <端口> root@<地址>:$WORK_DIR/outputs/yolov8n_person/weights/best.onnx ./"
echo ""
