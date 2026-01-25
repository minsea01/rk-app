#!/bin/bash
# ============================================================
#  CityPersons 行人检测训练
#  使用 AutoDL 预装 cityscapes + 下载 CityPersons 标注
# ============================================================

set -e

echo "============================================================"
echo "  CityPersons 行人检测训练 - 目标 90% mAP"
echo "============================================================"

WORK_DIR="/root/autodl-tmp/citypersons"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# 1. 安装依赖
echo "[1/5] 安装依赖..."
pip install ultralytics>=8.0.0 opencv-python-headless scipy --quiet

# 2. 检查 Cityscapes 图片
echo ""
echo "[2/5] 检查 Cityscapes..."

CITYSCAPES=""
for p in "/root/autodl-pub/cityscapes" "/root/autodl-pub/Cityscapes" "/data/cityscapes"; do
    if [ -d "$p" ]; then
        CITYSCAPES="$p"
        break
    fi
done

if [ -z "$CITYSCAPES" ]; then
    echo "未找到预装cityscapes，下载中..."
    mkdir -p datasets
    cd datasets
    # 使用学术加速下载
    pip install gdown --quiet
    gdown --fuzzy "https://drive.google.com/file/d/1DejVbpjiVsPSaco1-VYy6nUiGt7bxqW0/view" -O leftImg8bit_trainvaltest.zip
    unzip -q leftImg8bit_trainvaltest.zip
    CITYSCAPES="$WORK_DIR/datasets"
    cd "$WORK_DIR"
else
    echo "  找到: $CITYSCAPES"
fi

# 3. 下载 CityPersons 标注
echo ""
echo "[3/5] 下载 CityPersons 标注..."

mkdir -p annotations
cd annotations

if [ ! -f "anno_train.mat" ]; then
    # CityPersons 官方标注
    wget -q --show-progress "https://github.com/cvgroup-njust/CityPersons/raw/master/annotations/anno_train.mat" -O anno_train.mat || \
    wget -q --show-progress "https://bitbucket.org/shanshanzhang/citypersons/raw/master/annotations/anno_train.mat" -O anno_train.mat || \
    echo "下载anno_train.mat失败"
fi

if [ ! -f "anno_val.mat" ]; then
    wget -q --show-progress "https://github.com/cvgroup-njust/CityPersons/raw/master/annotations/anno_val.mat" -O anno_val.mat || \
    wget -q --show-progress "https://bitbucket.org/shanshanzhang/citypersons/raw/master/annotations/anno_val.mat" -O anno_val.mat || \
    echo "下载anno_val.mat失败"
fi

cd "$WORK_DIR"

# 4. 转换为 YOLO 格式
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
        return 0

    print(f"  处理 {split}...")

    # 加载 MAT 文件
    data = sio.loadmat(str(anno_file))
    annos = data[f'anno_{split}_aligned'][0] if f'anno_{split}_aligned' in data else data[f'anno_{split}'][0]

    count = 0
    for anno in annos:
        # 提取信息
        city_name = str(anno[0][0][0][0])
        img_name = str(anno[0][1][0][0])
        bboxes = anno[0][2]  # [class_label, x1, y1, w, h, instance_id, x1_vis, y1_vis, w_vis, h_vis]

        # 找图片
        img_path = None
        for subdir in ["leftImg8bit", "leftImg8bit_trainvaltest/leftImg8bit"]:
            p = cityscapes / subdir / split / city_name / f"{img_name}_leftImg8bit.png"
            if p.exists():
                img_path = p
                break

        if img_path is None or not img_path.exists():
            continue

        # 获取图片尺寸
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except:
            continue

        # 创建软链接
        link_path = output_dir / "images" / split / f"{city_name}_{img_name}.png"
        if not link_path.exists():
            os.symlink(img_path.resolve(), link_path)

        # 转换标注
        yolo_lines = []
        if bboxes is not None and len(bboxes) > 0:
            for box in bboxes:
                if len(box) < 5:
                    continue
                cls_label = int(box[0])
                # class 1 = pedestrian, 0 = ignore
                if cls_label != 1:
                    continue

                x1, y1, w, h = float(box[1]), float(box[2]), float(box[3]), float(box[4])

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

        if yolo_lines:
            label_file = output_dir / "labels" / split / f"{city_name}_{img_name}.txt"
            with open(label_file, 'w') as f:
                f.writelines(yolo_lines)
            count += 1

    return count

train_count = convert_split("train")
val_count = convert_split("val")

print(f"  train: {train_count} 张图片")
print(f"  val: {val_count} 张图片")

# 创建 YAML
yaml_content = f"""# CityPersons Dataset
path: {output_dir}
train: images/train
val: images/val

names:
  0: person

nc: 1
"""

with open(output_dir / "citypersons.yaml", 'w') as f:
    f.write(yaml_content)

print("  转换完成!")
PYEOF

# 5. 训练
echo ""
echo "[5/5] 开始训练..."

DATA_YAML="$WORK_DIR/datasets/citypersons_yolo/citypersons.yaml"

if [ ! -f "$DATA_YAML" ]; then
    echo "错误: 数据集配置不存在"
    echo "检查标注文件是否下载成功:"
    ls -la "$WORK_DIR/annotations/"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "警告: 未检测到GPU"

yolo detect train \
    model=yolov8n.pt \
    data="$DATA_YAML" \
    epochs=150 \
    imgsz=640 \
    batch=64 \
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

# 导出
echo ""
echo "导出 ONNX..."
yolo export model=outputs/yolov8n_citypersons/weights/best.pt format=onnx opset=12 simplify=True imgsz=640

echo ""
echo "============================================================"
echo "  完成!"
echo "============================================================"
echo ""
echo "下载:"
echo "scp -P <port> root@<host>:$WORK_DIR/outputs/yolov8n_citypersons/weights/best.pt ./"
echo "scp -P <port> root@<host>:$WORK_DIR/outputs/yolov8n_citypersons/weights/best.onnx ./"
