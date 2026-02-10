#!/bin/bash
# ============================================================
#  MOT17 行人检测训练 - 使用 AutoDL 预装数据集
#  无需下载！直接使用 /root/autodl-pub/mot17
# ============================================================

set -e

echo "============================================================"
echo "  MOT17 行人检测训练 - 目标 90% mAP"
echo "  使用 AutoDL 预装数据集，无需下载"
echo "============================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 工作目录
WORK_DIR="/root/autodl-tmp/mot17_training"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# ==================== 1. 安装依赖 ====================
echo "[1/4] 安装依赖..."
pip install ultralytics>=8.0.0 opencv-python-headless --quiet

# ==================== 2. 检查预装数据集 ====================
echo ""
echo "[2/4] 检查 MOT17 数据集..."

MOT17_ROOT=""
for path in "/root/autodl-pub/mot17" "/root/autodl-pub/MOT17" "/data/mot17"; do
    if [ -d "$path" ]; then
        MOT17_ROOT="$path"
        break
    fi
done

if [ -z "$MOT17_ROOT" ]; then
    echo "错误: 未找到 MOT17 数据集"
    echo "检查以下位置:"
    ls -la /root/autodl-pub/ 2>/dev/null || echo "  /root/autodl-pub 不存在"
    exit 1
fi

echo "  找到 MOT17: $MOT17_ROOT"
echo "  内容:"
ls -la "$MOT17_ROOT" | head -10

# ==================== 3. 转换为 YOLO 格式 ====================
echo ""
echo "[3/4] 转换为 YOLO 格式..."

python3 << PYEOF
import os
import json
import configparser
from pathlib import Path
from PIL import Image

mot17_root = Path("$MOT17_ROOT")
output_dir = Path("$WORK_DIR/datasets/mot17_yolo")

# 查找训练序列
train_dir = mot17_root / "train"
if not train_dir.exists():
    # 可能直接在根目录
    train_dir = mot17_root

print(f"  扫描: {train_dir}")

# 遍历所有序列
sequences = []
for seq_dir in sorted(train_dir.iterdir()):
    if seq_dir.is_dir() and (seq_dir / "gt" / "gt.txt").exists():
        sequences.append(seq_dir)
    elif seq_dir.is_dir() and (seq_dir / "det" / "det.txt").exists():
        sequences.append(seq_dir)

if not sequences:
    # 尝试查找 MOT17-XX-DPM 等格式
    for item in train_dir.iterdir():
        if item.is_dir() and "MOT17" in item.name:
            if (item / "gt" / "gt.txt").exists():
                sequences.append(item)

print(f"  找到 {len(sequences)} 个序列")

# 创建输出目录
(output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
(output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
(output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
(output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

total_images = 0
total_boxes = 0

for seq_dir in sequences:
    seq_name = seq_dir.name
    print(f"  处理: {seq_name}")

    # 读取序列信息
    seqinfo_file = seq_dir / "seqinfo.ini"
    img_dir = seq_dir / "img1"
    gt_file = seq_dir / "gt" / "gt.txt"

    if not img_dir.exists():
        print(f"    跳过: 无图片目录")
        continue

    # 获取图片尺寸
    img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not img_files:
        print(f"    跳过: 无图片")
        continue

    with Image.open(img_files[0]) as img:
        img_w, img_h = img.size

    # 读取 GT 标注
    # MOT 格式: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
    annotations = {}  # frame_id -> list of boxes

    if gt_file.exists():
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 7:
                    frame_id = int(parts[0])
                    # class: 1=pedestrian, 2=person on vehicle, 7=static person
                    cls = int(parts[7]) if len(parts) > 7 else 1
                    if cls not in [1, 2, 7]:  # 只要行人相关类别
                        continue

                    x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    conf = float(parts[6])

                    if conf <= 0:  # 忽略标记
                        continue

                    if frame_id not in annotations:
                        annotations[frame_id] = []
                    annotations[frame_id].append((x, y, w, h))

    # 决定训练/验证分割 (最后一个序列作为验证)
    split = "val" if seq_dir == sequences[-1] else "train"

    # 处理每帧
    for img_path in img_files:
        frame_id = int(img_path.stem)

        if frame_id not in annotations:
            continue

        boxes = annotations[frame_id]
        if not boxes:
            continue

        # 创建软链接
        link_name = f"{seq_name}_{img_path.name}"
        link_path = output_dir / "images" / split / link_name
        if not link_path.exists():
            os.symlink(img_path.resolve(), link_path)

        # 写 YOLO 标注
        label_file = output_dir / "labels" / split / f"{seq_name}_{img_path.stem}.txt"
        with open(label_file, 'w') as f:
            for (x, y, w, h) in boxes:
                # 转 YOLO 归一化格式
                x_center = (x + w/2) / img_w
                y_center = (y + h/2) / img_h
                width = w / img_w
                height = h / img_h

                # 边界检查
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0.001, min(1, width))
                height = max(0.001, min(1, height))

                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                total_boxes += 1

        total_images += 1

print(f"  总计: {total_images} 张图片, {total_boxes} 个行人标注")

# 创建 YAML 配置
yaml_content = f"""# MOT17 Pedestrian Dataset
path: {output_dir}
train: images/train
val: images/val

names:
  0: person

nc: 1
"""

with open(output_dir / "mot17.yaml", 'w') as f:
    f.write(yaml_content)

print("  转换完成!")
print(f"  配置文件: {output_dir}/mot17.yaml")
PYEOF

# 检查是否成功
if [ ! -f "$WORK_DIR/datasets/mot17_yolo/mot17.yaml" ]; then
    echo ""
    echo "MOT17 转换失败，尝试使用 Cityscapes..."

    # 使用 Cityscapes
    CITYSCAPES_ROOT=""
    for path in "/root/autodl-pub/cityscapes" "/root/autodl-pub/Cityscapes"; do
        if [ -d "$path" ]; then
            CITYSCAPES_ROOT="$path"
            break
        fi
    done

    if [ -n "$CITYSCAPES_ROOT" ]; then
        echo "  找到 Cityscapes: $CITYSCAPES_ROOT"
        # Cityscapes 转换逻辑...
    fi
fi

# ==================== 4. 开始训练 ====================
echo ""
echo "[4/4] 开始训练..."
echo "  Epochs: 150"
echo "  Batch: 64"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "警告: 未检测到 GPU"

DATA_YAML="$WORK_DIR/datasets/mot17_yolo/mot17.yaml"

if [ ! -f "$DATA_YAML" ]; then
    echo "错误: 数据集配置不存在"
    exit 1
fi

"$SCRIPT_DIR/train_runner.sh" \
  --profile baseline \
  --workdir "$WORK_DIR" \
  --model yolov8n.pt \
  --data "$DATA_YAML" \
  --epochs 150 \
  --imgsz 640 \
  --batch 64 \
  --device 0 \
  --project outputs \
  --name yolov8n_mot17 \
  --patience 50 \
  --save-period 10 \
  --workers 8 \
  --cache disk \
  --optimizer AdamW \
  --lr0 0.001 \
  --lrf 0.01

echo ""
echo "============================================================"
echo "  训练完成!"
echo "============================================================"

echo ""
echo "下载命令:"
echo "scp -P <端口> root@<地址>:$WORK_DIR/outputs/yolov8n_mot17/weights/best.pt ./"
echo "scp -P <端口> root@<地址>:$WORK_DIR/outputs/yolov8n_mot17/weights/best.onnx ./"
