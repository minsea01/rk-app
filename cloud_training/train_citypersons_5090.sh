#!/bin/bash
# ============================================================
#  CityPersons YOLOv8n 训练 - AutoDL RTX 5090 优化版
#  目标: mAP@0.5 >= 90%
# ============================================================
#
#  RTX 5090 配置:
#    - 32GB 显存 -> batch=192 可用
#    - 25 vCPU Xeon -> workers=12
#    - 90GB RAM -> cache=disk (安全)
#
#  使用方法:
#    chmod +x train_citypersons_5090.sh && ./train_citypersons_5090.sh
#
# ============================================================

set -e

echo "============================================================"
echo "  YOLOv8n CityPersons 行人检测 - RTX 5090 优化"
echo "  目标: 90% mAP | 预计 2-3 小时"
echo "============================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ==================== 配置参数 ====================
EPOCHS=${EPOCHS:-200}
BATCH=${BATCH:-192}        # RTX 5090 32GB 可用 192
IMGSZ=${IMGSZ:-640}
WORKERS=${WORKERS:-12}
CACHE=${CACHE:-disk}       # disk 更安全，避免 RAM 溢出
LR0=${LR0:-0.001}

WORK_DIR="/root/autodl-tmp/citypersons"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# ==================== 1. 环境准备 ====================
echo "[1/5] 安装依赖..."
pip install ultralytics>=8.0.0 opencv-python-headless scipy pycocotools --quiet

# 显示 GPU 信息
echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "警告: 未检测到 GPU"

# ==================== 2. 检查 Cityscapes ====================
echo ""
echo "[2/5] 检查 Cityscapes 数据集..."

CITYSCAPES=""
for p in "/root/autodl-pub/cityscapes" "/root/autodl-pub/Cityscapes" "/root/autodl-tmp/cityscapes" "/data/cityscapes"; do
    if [ -d "$p" ]; then
        CITYSCAPES="$p"
        echo "  找到: $CITYSCAPES"
        break
    fi
done

if [ -z "$CITYSCAPES" ]; then
    echo "  未找到预装 Cityscapes"
    echo ""
    echo "  选项 1: 使用 AutoDL 公共数据集 (推荐)"
    echo "    - 在控制台挂载 cityscapes 数据集"
    echo ""
    echo "  选项 2: 切换到 COCO Person 训练"
    echo "    ./one_click_train.sh"
    echo ""
    exit 1
fi

# ==================== 3. 下载 CityPersons 标注 ====================
echo ""
echo "[3/5] 下载 CityPersons 标注..."

mkdir -p annotations
cd annotations

# 尝试多个下载源
if [ ! -f "anno_train.mat" ]; then
    echo "  下载 anno_train.mat..."
    wget -q --show-progress "https://github.com/cvgroup-njust/CityPersons/raw/master/annotations/anno_train.mat" -O anno_train.mat 2>/dev/null || \
    wget -q --show-progress "https://raw.githubusercontent.com/CharlesShang/Detectron-OHEM/master/data/citypersons/annotations/anno_train.mat" -O anno_train.mat 2>/dev/null || \
    echo "  警告: 下载 anno_train.mat 失败"
fi

if [ ! -f "anno_val.mat" ]; then
    echo "  下载 anno_val.mat..."
    wget -q --show-progress "https://github.com/cvgroup-njust/CityPersons/raw/master/annotations/anno_val.mat" -O anno_val.mat 2>/dev/null || \
    wget -q --show-progress "https://raw.githubusercontent.com/CharlesShang/Detectron-OHEM/master/data/citypersons/annotations/anno_val.mat" -O anno_val.mat 2>/dev/null || \
    echo "  警告: 下载 anno_val.mat 失败"
fi

cd "$WORK_DIR"

# ==================== 4. 转换为 YOLO 格式 ====================
echo ""
echo "[4/5] 转换为 YOLO 格式..."

python3 << PYEOF
import os
import scipy.io as sio
from pathlib import Path
from PIL import Image

cityscapes = Path("$CITYSCAPES")
work_dir = Path("$WORK_DIR")
output_dir = work_dir / "datasets" / "citypersons_yolo"

# 创建目录
for split in ["train", "val"]:
    (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

def convert_split(split):
    anno_file = work_dir / "annotations" / f"anno_{split}.mat"
    if not anno_file.exists():
        print(f"  警告: {anno_file} 不存在")
        return 0, 0

    print(f"  处理 {split}...")

    # 加载 MAT 文件
    data = sio.loadmat(str(anno_file))

    # 尝试不同的 key
    anno_key = None
    for key in [f'anno_{split}_aligned', f'anno_{split}']:
        if key in data:
            anno_key = key
            break

    if anno_key is None:
        print(f"  错误: 无法解析 {anno_file}")
        return 0, 0

    annos = data[anno_key][0]

    count = 0
    total_persons = 0

    for anno in annos:
        try:
            # 提取信息
            city_name = str(anno[0][0][0][0])
            img_name = str(anno[0][1][0][0])
            bboxes = anno[0][2]
        except:
            continue

        # 找图片
        img_path = None
        for subdir in ["leftImg8bit", "leftImg8bit_trainvaltest/leftImg8bit"]:
            p = cityscapes / subdir / split / city_name / f"{img_name}_leftImg8bit.png"
            if p.exists():
                img_path = p
                break

        if img_path is None:
            continue

        # 获取图片尺寸
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except:
            continue

        # 创建软链接 (避免复制大文件)
        link_path = output_dir / "images" / split / f"{city_name}_{img_name}.png"
        if not link_path.exists():
            try:
                os.symlink(img_path.resolve(), link_path)
            except:
                pass

        # 转换标注
        yolo_lines = []
        if bboxes is not None and len(bboxes) > 0:
            for box in bboxes:
                if len(box) < 5:
                    continue
                cls_label = int(box[0])
                # class 1 = pedestrian (忽略 0=ignore, 2=rider 等)
                if cls_label != 1:
                    continue

                x1, y1, w, h = float(box[1]), float(box[2]), float(box[3]), float(box[4])

                # 过滤太小的目标
                if w < 10 or h < 20:
                    continue

                # 转 YOLO 格式
                x_center = (x1 + w/2) / img_w
                y_center = (y1 + h/2) / img_h
                width = w / img_w
                height = h / img_h

                # 边界检查
                if width <= 0 or height <= 0:
                    continue
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0.001, min(1, width))
                height = max(0.001, min(1, height))

                yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                total_persons += 1

        if yolo_lines:
            label_file = output_dir / "labels" / split / f"{city_name}_{img_name}.txt"
            with open(label_file, 'w') as f:
                f.writelines(yolo_lines)
            count += 1

    return count, total_persons

train_count, train_persons = convert_split("train")
val_count, val_persons = convert_split("val")

print(f"  train: {train_count} 张图片, {train_persons} 个行人")
print(f"  val: {val_count} 张图片, {val_persons} 个行人")

# 创建 YAML 配置
yaml_content = f"""# CityPersons Dataset (YOLO format)
path: {output_dir}
train: images/train
val: images/val

names:
  0: person

nc: 1
"""

with open(output_dir / "citypersons.yaml", 'w') as f:
    f.write(yaml_content)

print("  YAML 配置已创建")
PYEOF

# ==================== 5. 开始训练 ====================
echo ""
echo "[5/5] 开始训练..."
echo "  Epochs: $EPOCHS"
echo "  Batch: $BATCH (RTX 5090 优化)"
echo "  Image Size: $IMGSZ"
echo "  Cache: $CACHE"
echo ""

DATA_YAML="$WORK_DIR/datasets/citypersons_yolo/citypersons.yaml"

if [ ! -f "$DATA_YAML" ]; then
    echo "错误: 数据集配置不存在"
    echo "检查标注文件:"
    ls -la "$WORK_DIR/annotations/" 2>/dev/null || echo "  annotations/ 目录不存在"
    exit 1
fi

"$SCRIPT_DIR/train_runner.sh" \
  --profile baseline \
  --workdir "$WORK_DIR" \
  --model yolov8n.pt \
  --data "$DATA_YAML" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --device 0 \
  --project outputs \
  --name yolov8n_citypersons \
  --patience 50 \
  --save-period 20 \
  --workers "$WORKERS" \
  --cache "$CACHE" \
  --optimizer AdamW \
  --lr0 "$LR0" \
  --lrf 0.01 \
  --warmup-epochs 5 \
  --extra "shear=2.0"

echo ""
echo "============================================================"
echo "  训练完成!"
echo "============================================================"
echo ""
echo "模型位置:"
echo "  PT:   $WORK_DIR/outputs/yolov8n_citypersons/weights/best.pt"
echo "  ONNX: $WORK_DIR/outputs/yolov8n_citypersons/weights/best.onnx"
echo ""
echo "下载命令 (本地执行):"
echo "  scp -P <端口> root@<地址>:$WORK_DIR/outputs/yolov8n_citypersons/weights/best.pt ./artifacts/models/"
echo "  scp -P <端口> root@<地址>:$WORK_DIR/outputs/yolov8n_citypersons/weights/best.onnx ./artifacts/models/"
echo ""
