#!/bin/bash
# AutoDL 4090 环境配置脚本
# 选择镜像: PyTorch 2.0.0 + Python 3.10 + CUDA 11.8

set -e

echo "========================================="
echo "AutoDL YOLOv8n 行人检测训练环境配置"
echo "========================================="

# 1. 安装依赖
echo "[1/4] 安装Python依赖..."
pip install ultralytics>=8.0.0 -q
pip install albumentations>=1.0.0 -q
pip install gdown -q  # 用于下载CityPersons

# 2. 创建项目目录
echo "[2/4] 创建项目结构..."
mkdir -p ~/pedestrian_training/{datasets,models,outputs}
cd ~/pedestrian_training

# 3. 下载YOLOv8n预训练权重
echo "[3/4] 下载YOLOv8n预训练权重..."
if [ ! -f models/yolov8n.pt ]; then
    wget -q https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt -O models/yolov8n.pt
    echo "  -> yolov8n.pt 下载完成"
else
    echo "  -> yolov8n.pt 已存在"
fi

# 4. 提示下一步
echo "[4/4] 环境配置完成!"
echo ""
echo "========================================="
echo "下一步操作:"
echo "1. 运行 download_citypersons.sh 下载数据集"
echo "2. 运行 train.sh 开始训练"
echo "========================================="
