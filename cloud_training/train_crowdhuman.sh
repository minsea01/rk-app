#!/bin/bash
# ============================================================
#  CrowdHuman 行人检测训练 - 目标 90% mAP
#  CrowdHuman 是专门的人群检测数据集，比 COCO 效果好很多
# ============================================================

set -e

echo "============================================================"
echo "  CrowdHuman 行人检测训练 - 目标 90% mAP"
echo "============================================================"
echo ""

# 工作目录
WORK_DIR="/root/autodl-tmp/crowdhuman_training"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# ==================== 1. 安装依赖 ====================
echo "[1/5] 安装依赖..."
pip install ultralytics>=8.0.0 opencv-python-headless gdown --quiet

# ==================== 2. 下载 CrowdHuman ====================
echo ""
echo "[2/5] 下载 CrowdHuman 数据集..."

mkdir -p datasets/crowdhuman
cd datasets/crowdhuman

# CrowdHuman 数据集下载 (使用 gdown 从 Google Drive)
# 官方链接需要申请，这里用公开镜像

if [ ! -f "annotation_train.odgt" ]; then
    echo "  下载标注文件..."
    # 标注文件 (小文件，直接下载)
    wget -q --show-progress -O annotation_train.odgt \
        "https://raw.githubusercontent.com/Shaoqing/CrowdHuman-annotations/main/annotation_train.odgt" 2>/dev/null || \
    gdown --fuzzy "https://drive.google.com/file/d/1UUTea5mYqvlUObsC1Z8CFldHJAtLtMX3/view" -O annotation_train.odgt 2>/dev/null || \
    echo "  请手动下载 annotation_train.odgt"

    wget -q --show-progress -O annotation_val.odgt \
        "https://raw.githubusercontent.com/Shaoqing/CrowdHuman-annotations/main/annotation_val.odgt" 2>/dev/null || \
    gdown --fuzzy "https://drive.google.com/file/d/10WIRwu8ju8GRLuCkZ_vT6hnNxs5ptwoL/view" -O annotation_val.odgt 2>/dev/null || \
    echo "  请手动下载 annotation_val.odgt"
fi

# 下载图片 (如果不存在)
if [ ! -d "Images" ] || [ $(ls Images 2>/dev/null | wc -l) -lt 1000 ]; then
    echo ""
    echo "  下载 CrowdHuman 图片 (约 10GB)..."
    echo "  这可能需要 20-30 分钟..."

    # 使用 HuggingFace 镜像 (更快)
    pip install huggingface_hub --quiet

    python3 << 'PYEOF'
import os
from huggingface_hub import hf_hub_download, snapshot_download

try:
    # 尝试从 HuggingFace 下载
    print("  从 HuggingFace 下载...")
    snapshot_download(
        repo_id="zhiqings/CrowdHuman",
        repo_type="dataset",
        local_dir=".",
        allow_patterns=["*.zip", "*.odgt"]
    )
except Exception as e:
    print(f"  HuggingFace 下载失败: {e}")
    print("  请手动下载 CrowdHuman 数据集")
    print("  官网: https://www.crowdhuman.org/")
PYEOF

    # 解压图片
    for f in CrowdHuman_train*.zip CrowdHuman_val*.zip; do
        if [ -f "$f" ]; then
            echo "  解压 $f..."
            unzip -q -o "$f"
        fi
    done
fi

cd "$WORK_DIR"

# ==================== 3. 转换为 YOLO 格式 ====================
echo ""
echo "[3/5] 转换为 YOLO 格式..."

python3 << 'PYEOF'
import json
import os
from pathlib import Path
from PIL import Image

dataset_dir = Path("datasets/crowdhuman")
output_dir = Path("datasets/crowdhuman_yolo")

for split in ["train", "val"]:
    ann_file = dataset_dir / f"annotation_{split}.odgt"
    if not ann_file.exists():
        print(f"  警告: {ann_file} 不存在")
        continue

    print(f"  处理 {split}...")

    img_out = output_dir / "images" / split
    lbl_out = output_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(ann_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            img_id = data['ID']

            # 找图片
            img_path = None
            for ext in ['.jpg', '.png', '.jpeg']:
                for img_dir in ['Images', 'images', split]:
                    p = dataset_dir / img_dir / f"{img_id}{ext}"
                    if p.exists():
                        img_path = p
                        break
                if img_path:
                    break

            if not img_path or not img_path.exists():
                continue

            # 获取图片尺寸
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except:
                continue

            # 创建软链接
            link_path = img_out / f"{img_id}.jpg"
            if not link_path.exists():
                os.symlink(img_path.resolve(), link_path)

            # 转换标注
            yolo_lines = []
            if 'gtboxes' in data:
                for box in data['gtboxes']:
                    if box.get('tag') == 'person':
                        # fbox: [x, y, w, h] 全身框
                        x, y, w, h = box['fbox']

                        # 转 YOLO 格式
                        x_center = (x + w/2) / img_w
                        y_center = (y + h/2) / img_h
                        width = w / img_w
                        height = h / img_h

                        # 边界检查
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0.001, min(1, width))
                        height = max(0.001, min(1, height))

                        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            if yolo_lines:
                with open(lbl_out / f"{img_id}.txt", 'w') as f:
                    f.writelines(yolo_lines)
                count += 1

    print(f"    {split}: {count} 张图片")

# 创建 YAML 配置
yaml_content = f"""# CrowdHuman Dataset
path: {output_dir.resolve()}
train: images/train
val: images/val

names:
  0: person

nc: 1
"""

with open(output_dir / "crowdhuman.yaml", 'w') as f:
    f.write(yaml_content)

print("  转换完成!")
PYEOF

# ==================== 4. 训练 ====================
echo ""
echo "[4/5] 开始训练..."
echo "  Epochs: 150"
echo "  Batch: 64"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "警告: 未检测到 GPU"

yolo detect train \
    model=yolov8n.pt \
    data=datasets/crowdhuman_yolo/crowdhuman.yaml \
    epochs=150 \
    imgsz=640 \
    batch=64 \
    device=0 \
    project=outputs \
    name=yolov8n_crowdhuman \
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
    warmup_epochs=3 \
    mosaic=1.0 \
    mixup=0.15 \
    copy_paste=0.1 \
    degrees=5.0 \
    translate=0.1 \
    scale=0.5 \
    fliplr=0.5 \
    workers=8 \
    cache=disk \
    amp=True

# ==================== 5. 导出 ====================
echo ""
echo "[5/5] 导出 ONNX..."

yolo export \
    model=outputs/yolov8n_crowdhuman/weights/best.pt \
    format=onnx \
    opset=12 \
    simplify=True \
    imgsz=640

# 结果
echo ""
echo "============================================================"
echo "  训练完成!"
echo "============================================================"

python3 << 'PYEOF'
import csv
import os

results_file = "outputs/yolov8n_crowdhuman/results.csv"
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
                        else:
                            print(f"⚠️ 差 {(0.90-val)*100:.1f}%，建议增加 epochs")
                    except:
                        pass
                    break

pt_file = "outputs/yolov8n_crowdhuman/weights/best.pt"
onnx_file = "outputs/yolov8n_crowdhuman/weights/best.onnx"
if os.path.exists(onnx_file):
    size = os.path.getsize(onnx_file) / 1024 / 1024
    print(f"\nONNX 大小: {size:.1f} MB")
    print(f"RKNN INT8 预估: {size*0.4:.1f} MB")
PYEOF

echo ""
echo "下载命令:"
echo "scp -P 18574 root@connect.westd.seetacloud.com:$WORK_DIR/outputs/yolov8n_crowdhuman/weights/best.pt ./"
echo "scp -P 18574 root@connect.westd.seetacloud.com:$WORK_DIR/outputs/yolov8n_crowdhuman/weights/best.onnx ./"
