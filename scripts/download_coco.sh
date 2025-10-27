#!/bin/bash
# COCO 2017数据集下载脚本

set -e

DOWNLOAD_DIR=${1:-"datasets/coco_raw"}
echo "Downloading COCO 2017 dataset to: $DOWNLOAD_DIR"

mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

# 下载地址
TRAIN_IMGS="http://images.cocodataset.org/zips/train2017.zip"
VAL_IMGS="http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

echo "============================================"
echo "Downloading COCO 2017 Dataset"
echo "============================================"

# 下载训练集图像 (约18GB)
if [ ! -f "train2017.zip" ]; then
    echo "[1/3] Downloading training images (18GB)..."
    wget --continue "$TRAIN_IMGS"
else
    echo "[1/3] train2017.zip already exists, skipping download"
fi

# 下载验证集图像 (约1GB)
if [ ! -f "val2017.zip" ]; then
    echo "[2/3] Downloading validation images (1GB)..."
    wget --continue "$VAL_IMGS"
else
    echo "[2/3] val2017.zip already exists, skipping download"
fi

# 下载标注文件 (约241MB)
if [ ! -f "annotations_trainval2017.zip" ]; then
    echo "[3/3] Downloading annotations (241MB)..."
    wget --continue "$ANNOTATIONS"
else
    echo "[3/3] annotations_trainval2017.zip already exists, skipping download"
fi

echo ""
echo "============================================"
echo "Extracting files..."
echo "============================================"

# 解压文件
if [ ! -d "train2017" ]; then
    echo "Extracting training images..."
    unzip -q train2017.zip
fi

if [ ! -d "val2017" ]; then
    echo "Extracting validation images..."
    unzip -q val2017.zip
fi

if [ ! -d "annotations" ]; then
    echo "Extracting annotations..."
    unzip -q annotations_trainval2017.zip
fi

echo ""
echo "============================================"
echo "✓ Download completed!"
echo "============================================"
echo "Dataset structure:"
tree -L 1 -d .

echo ""
echo "Next steps:"
echo "  1. Process COCO person dataset:"
echo "     python tools/prepare_coco_person.py --coco-dir $DOWNLOAD_DIR --output-dir datasets/coco_person"
echo ""
echo "  2. (Optional) Copy images to output directory:"
echo "     python tools/prepare_coco_person.py --coco-dir $DOWNLOAD_DIR --output-dir datasets/coco_person --copy-images"
