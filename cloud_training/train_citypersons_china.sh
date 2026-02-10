#!/bin/bash
# ============================================================
#  CityPersons YOLOv8n 训练 - 国内网络优化版
#  RTX 5090 (32GB) | 目标 mAP@0.5 >= 90%
# ============================================================
#
#  特点:
#    1. 多镜像源自动切换 (Gitee/ghproxy/jsdelivr)
#    2. 支持本地预下载标注文件
#    3. 断点续训
#    4. RTX 5090 优化参数
#
#  使用方法:
#    方案1 (推荐): 本地先下载标注，一起上传
#      cd cloud_training/citypersons_data
#      python3 download_annotations.py
#      cd .. && tar -czvf citypersons_train.tar.gz citypersons_data/ train_citypersons_china.sh
#      scp -P<端口> citypersons_train.tar.gz root@<地址>:/root/
#
#    方案2: 直接在 AutoDL 运行 (需要网络)
#      ./train_citypersons_china.sh
#
# ============================================================

set -e

echo "============================================================"
echo "  YOLOv8n CityPersons 行人检测 - 国内优化版"
echo "  RTX 5090 | 目标: 90% mAP"
echo "============================================================"
echo ""

# ==================== 配置参数 ====================
EPOCHS=${EPOCHS:-200}
BATCH=${BATCH:-192}        # RTX 5090: 192, RTX 4090: 128
IMGSZ=${IMGSZ:-640}
WORKERS=${WORKERS:-12}
CACHE=${CACHE:-disk}
LR0=${LR0:-0.001}

WORK_DIR="/root/autodl-tmp/citypersons"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# ==================== 1. 环境准备 ====================
echo "[1/6] 安装依赖..."

# 使用清华源加速
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || true

pip install ultralytics>=8.0.0 opencv-python-headless scipy pycocotools --quiet

echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "警告: 未检测到 GPU"

# ==================== 2. 检查 Cityscapes ====================
echo ""
echo "[2/6] 检查 Cityscapes 数据集..."

CITYSCAPES=""
for p in "/root/autodl-pub/cityscapes" \
         "/root/autodl-pub/Cityscapes" \
         "/root/autodl-tmp/cityscapes" \
         "/data/cityscapes" \
         "/root/shared-storage/cityscapes"; do
    if [ -d "$p/leftImg8bit" ]; then
        CITYSCAPES="$p"
        echo "  找到: $CITYSCAPES"
        # 统计图片数量
        TRAIN_COUNT=$(find "$CITYSCAPES/leftImg8bit/train" -name "*.png" 2>/dev/null | wc -l)
        VAL_COUNT=$(find "$CITYSCAPES/leftImg8bit/val" -name "*.png" 2>/dev/null | wc -l)
        echo "  train: $TRAIN_COUNT 张, val: $VAL_COUNT 张"
        break
    fi
done

if [ -z "$CITYSCAPES" ]; then
    echo "  未找到 Cityscapes 数据集!"
    echo ""
    echo "  解决方案:"
    echo "  1. 在 AutoDL 控制台 -> 数据集 -> 挂载 'cityscapes'"
    echo "  2. 或使用 COCO Person (one_click_train.sh)"
    echo ""
    exit 1
fi

# ==================== 3. 获取 CityPersons 标注 ====================
echo ""
echo "[3/6] 获取 CityPersons 标注..."

mkdir -p annotations

# 检查是否有本地预下载的标注
LOCAL_ANNO="$SCRIPT_DIR/citypersons_data"
if [ -f "$LOCAL_ANNO/anno_train.mat" ] && [ -f "$LOCAL_ANNO/anno_val.mat" ]; then
    echo "  使用本地预下载的标注文件"
    cp "$LOCAL_ANNO/anno_train.mat" annotations/
    cp "$LOCAL_ANNO/anno_val.mat" annotations/
else
    echo "  从网络下载标注..."

    # 下载函数
    download_with_retry() {
        local filename=$1
        local dest="annotations/$filename"

        if [ -f "$dest" ]; then
            echo "  $filename 已存在"
            return 0
        fi

        # 镜像源列表
        local urls=(
            # Gitee 镜像 (国内快)
            "https://gitee.com/mirrors_cvgroup-njust/CityPersons/raw/master/annotations/$filename"
            # ghproxy 代理
            "https://ghproxy.com/https://github.com/cvgroup-njust/CityPersons/raw/master/annotations/$filename"
            # jsdelivr CDN
            "https://cdn.jsdelivr.net/gh/cvgroup-njust/CityPersons@master/annotations/$filename"
            # GitHub 直连 (备用)
            "https://github.com/cvgroup-njust/CityPersons/raw/master/annotations/$filename"
            # 备用仓库
            "https://raw.githubusercontent.com/CharlesShang/Detectron-OHEM/master/data/citypersons/annotations/$filename"
        )

        for url in "${urls[@]}"; do
            echo "  尝试: ${url:0:50}..."
            if wget -q --timeout=30 --tries=2 "$url" -O "$dest" 2>/dev/null; then
                # 验证文件大小
                local size=$(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest" 2>/dev/null)
                if [ "$size" -gt 100000 ]; then
                    echo "  成功! ($(($size/1024)) KB)"
                    return 0
                else
                    rm -f "$dest"
                fi
            fi
        done

        echo "  $filename 下载失败!"
        return 1
    }

    download_with_retry "anno_train.mat" || {
        echo ""
        echo "  标注文件下载失败，请手动下载:"
        echo "  1. 访问 https://github.com/cvgroup-njust/CityPersons/tree/master/annotations"
        echo "  2. 下载 anno_train.mat 和 anno_val.mat"
        echo "  3. 上传到 $WORK_DIR/annotations/"
        exit 1
    }

    download_with_retry "anno_val.mat" || exit 1
fi

# 验证标注文件
echo "  验证标注文件..."
python3 << 'PYEOF'
import scipy.io as sio
import sys

for name in ["anno_train", "anno_val"]:
    try:
        data = sio.loadmat(f"annotations/{name}.mat")
        found = False
        for key in [name, f"{name}_aligned"]:
            if key in data:
                count = len(data[key][0])
                print(f"  {name}.mat: {count} 条记录")
                found = True
                break
        if not found:
            print(f"  警告: {name}.mat 格式异常")
            sys.exit(1)
    except Exception as e:
        print(f"  错误: 无法读取 {name}.mat - {e}")
        sys.exit(1)
print("  标注验证通过")
PYEOF

# ==================== 4. 转换为 YOLO 格式 ====================
echo ""
echo "[4/6] 转换为 YOLO 格式..."

python3 << PYEOF
import os
import scipy.io as sio
from pathlib import Path
from PIL import Image
import warnings
import numpy as np
warnings.filterwarnings('ignore')

cityscapes = Path("$CITYSCAPES")
work_dir = Path("$WORK_DIR")
output_dir = work_dir / "datasets" / "citypersons_yolo"

# 创建目录
for split in ["train", "val"]:
    (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

def find_image(cityscapes, split, city_name, img_name):
    """查找图片路径 - 支持多种命名格式"""
    # img_name 可能是完整文件名或不带后缀
    base_name = img_name.replace("_leftImg8bit.png", "").replace("_leftImg8bit", "")

    patterns = [
        cityscapes / "leftImg8bit" / split / city_name / f"{base_name}_leftImg8bit.png",
        cityscapes / "leftImg8bit" / split / city_name / img_name,
        cityscapes / "leftImg8bit_trainvaltest" / "leftImg8bit" / split / city_name / f"{base_name}_leftImg8bit.png",
    ]
    for p in patterns:
        if p.exists():
            return p
    return None

def convert_split(split):
    anno_file = work_dir / "annotations" / f"anno_{split}.mat"
    if not anno_file.exists():
        print(f"  警告: {anno_file} 不存在")
        return 0, 0

    print(f"  处理 {split}...")

    data = sio.loadmat(str(anno_file))

    # 查找正确的 key
    anno_key = None
    for key in [f'anno_{split}_aligned', f'anno_{split}']:
        if key in data:
            anno_key = key
            break

    if anno_key is None:
        print(f"  错误: 无法解析标注文件")
        return 0, 0

    annos = data[anno_key][0]

    count = 0
    total_persons = 0
    skipped = 0

    for anno in annos:
        try:
            # CityPersons 标注是 structured array: (cityname, im_name, bbs)
            record = anno[0, 0]  # 获取结构体

            # 提取字段 - 支持多种格式
            if hasattr(record, 'dtype') and record.dtype.names:
                # Structured array with named fields
                city_name = str(record['cityname'][0])
                img_name = str(record['im_name'][0])
                bboxes = record['bbs']
            else:
                # Nested array format
                city_name = str(record[0][0])
                img_name = str(record[1][0])
                bboxes = record[2]
        except Exception as e:
            skipped += 1
            continue

        img_path = find_image(cityscapes, split, city_name, img_name)
        if img_path is None:
            skipped += 1
            continue

        # 获取图片尺寸 (Cityscapes 标准: 2048x1024)
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except:
            # 使用默认尺寸
            img_w, img_h = 2048, 1024

        # 创建软链接
        base_name = img_name.replace("_leftImg8bit.png", "").replace("_leftImg8bit", "")
        link_name = f"{city_name}_{base_name}.png"
        link_path = output_dir / "images" / split / link_name
        if not link_path.exists():
            try:
                os.symlink(img_path.resolve(), link_path)
            except FileExistsError:
                pass

        # 转换标注
        yolo_lines = []
        if bboxes is not None and len(bboxes) > 0:
            for box in bboxes:
                if len(box) < 5:
                    continue

                cls_label = int(box[0])
                # 1 = pedestrian, 忽略其他类别
                if cls_label != 1:
                    continue

                x1, y1, w, h = float(box[1]), float(box[2]), float(box[3]), float(box[4])

                # 过滤太小的目标 (行人检测常用阈值)
                if h < 50:  # 高度小于 50 像素忽略
                    continue

                # YOLO 格式
                x_center = (x1 + w/2) / img_w
                y_center = (y1 + h/2) / img_h
                width = w / img_w
                height = h / img_h

                # 边界检查
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0.001, min(1, width))
                height = max(0.001, min(1, height))

                if width > 0 and height > 0:
                    yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    total_persons += 1

        if yolo_lines:
            label_file = output_dir / "labels" / split / f"{city_name}_{img_name}.txt"
            with open(label_file, 'w') as f:
                f.writelines(yolo_lines)
            count += 1

    if skipped > 0:
        print(f"    跳过 {skipped} 条无法匹配的记录")

    return count, total_persons

train_count, train_persons = convert_split("train")
val_count, val_persons = convert_split("val")

print(f"  train: {train_count} 张图片, {train_persons} 个行人")
print(f"  val: {val_count} 张图片, {val_persons} 个行人")

if train_count == 0:
    print("  错误: 没有生成训练数据!")
    print("  请检查 Cityscapes 数据集路径是否正确")
    exit(1)

# 创建 YAML 配置
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

print("  YAML 配置已创建")
PYEOF

# ==================== 5. 开始训练 ====================
echo ""
echo "[5/6] 开始训练..."
echo "  Epochs: $EPOCHS"
echo "  Batch: $BATCH"
echo "  ImgSize: $IMGSZ"
echo ""

DATA_YAML="$WORK_DIR/datasets/citypersons_yolo/citypersons.yaml"

if [ ! -f "$DATA_YAML" ]; then
    echo "错误: 数据集配置不存在"
    exit 1
fi

# 检查是否有之前的训练
RESUME_MODEL="yolov8n.pt"
if [ -f "outputs/yolov8n_citypersons/weights/last.pt" ]; then
    echo "  发现之前的训练，从断点继续..."
    RESUME_MODEL="outputs/yolov8n_citypersons/weights/last.pt"
fi

"$SCRIPT_DIR/train_runner.sh" \
  --profile baseline \
  --workdir "$WORK_DIR" \
  --model "$RESUME_MODEL" \
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
echo "下载命令:"
echo "  scp -P <端口> root@<地址>:$WORK_DIR/outputs/yolov8n_citypersons/weights/best.pt ./"
echo "  scp -P <端口> root@<地址>:$WORK_DIR/outputs/yolov8n_citypersons/weights/best.onnx ./"
echo ""
