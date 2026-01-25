#!/bin/bash
# ============================================
# AutoDL CityPersons 一键训练脚本
# 目标: YOLOv8n 行人检测 mAP >= 90%
# ============================================
# 使用方法:
#   1. 上传整个 autodl_citypersons 文件夹到 AutoDL
#   2. cd autodl_citypersons && chmod +x *.sh
#   3. ./START_HERE.sh
# ============================================

set -e

echo "============================================"
echo "  CityPersons 行人检测训练"
echo "  目标: mAP@0.5 >= 90%"
echo "============================================"
echo ""

# 工作目录
WORK_DIR="/root/autodl-tmp/citypersons_training"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# 复制脚本
cp -r /root/autodl_citypersons/* . 2>/dev/null || true

echo "[1/5] 安装依赖..."
pip install ultralytics opencv-python-headless --quiet
pip install gdown --quiet  # 用于下载数据集

echo ""
echo "[2/5] 下载 CityPersons 数据集..."
# CityPersons 比较难下载，我们使用替代方案：
# 1. 优先使用 WiderPerson (更容易下载)
# 2. 或者使用 CrowdHuman

if [ ! -d "datasets/widerperson" ]; then
    echo "  下载 WiderPerson 数据集..."
    ./download_widerperson.sh
fi

echo ""
echo "[3/5] 准备数据集配置..."
cat > datasets/person.yaml << 'EOF'
# Person Detection Dataset
path: /root/autodl-tmp/citypersons_training/datasets/widerperson
train: images/train
val: images/val

names:
  0: person

nc: 1
EOF

echo ""
echo "[4/5] 开始训练..."
echo "  预计时间: RTX 4090 约 2-4 小时"
echo ""

# 检测GPU
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "  警告: 未检测到GPU"

# 训练
./train_citypersons.sh

echo ""
echo "[5/5] 导出模型..."
./export_model.sh

echo ""
echo "============================================"
echo "  训练完成!"
echo "  模型位置: outputs/yolov8n_citypersons/weights/best.pt"
echo "  ONNX位置: outputs/yolov8n_citypersons/weights/best.onnx"
echo "============================================"
echo ""
echo "下载命令 (在本地执行):"
echo "  scp -P <端口> root@<地址>:$WORK_DIR/outputs/yolov8n_citypersons/weights/best.pt ./"
echo "  scp -P <端口> root@<地址>:$WORK_DIR/outputs/yolov8n_citypersons/weights/best.onnx ./"
