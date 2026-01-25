#!/bin/bash
# WiderPerson 数据集下载脚本
# 专门为行人检测设计的大型数据集
# 8000 训练图像 + 1000 验证图像

set -e

cd ~/pedestrian_training
mkdir -p datasets/widerperson
cd datasets/widerperson

echo "========================================="
echo "WiderPerson 数据集下载"
echo "========================================="

# 方案1: 从百度网盘镜像下载 (需要手动)
echo ""
echo "[方案1] 百度网盘下载 (推荐中国用户):"
echo "  链接: https://pan.baidu.com/s/1kkugS8oAXVAh6qrgKYSxYA"
echo "  密码: p9ou"
echo ""

# 方案2: 直接从GitHub release下载YOLO格式版本
echo "[方案2] 使用预转换YOLO格式数据集..."

# 尝试从huggingface下载
if command -v wget &> /dev/null; then
    echo "尝试从Huggingface下载WiderPerson YOLO格式..."

    # 创建目录结构
    mkdir -p {train,val}/{images,labels}

    # 检查是否已存在
    if [ -f "train/images/.downloaded" ]; then
        echo "数据集已存在，跳过下载"
    else
        echo "正在下载..."

        # 从Roboflow下载pedestrian detection数据集
        pip install roboflow -q

        python3 << 'PYEOF'
from roboflow import Roboflow
import os

print("使用Roboflow下载行人检测数据集...")

# 使用公开的行人检测数据集
# 方案A: 使用PPE Detection数据集(包含person类)
# 方案B: 使用Pedestrian Detection数据集

try:
    # 尝试下载一个高质量的行人数据集
    rf = Roboflow(api_key="")  # 公开数据集不需要API key

    # 使用pedestrian-detection-dataset
    project = rf.workspace("roboflow-universe-projects").project("pedestrian-detection-12keo")
    dataset = project.version(1).download("yolov8")

    print(f"数据集下载到: {dataset.location}")

except Exception as e:
    print(f"Roboflow下载失败: {e}")
    print("请使用方案1手动下载")
PYEOF

        touch train/images/.downloaded 2>/dev/null || true
    fi
fi

# 方案3: 使用Penn-Fudan数据集 (小但高质量)
echo ""
echo "[方案3] Penn-Fudan行人数据集 (小型高质量)..."

mkdir -p pennfudan
cd pennfudan

if [ ! -f ".downloaded" ]; then
    wget -q https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip -O PennFudanPed.zip 2>/dev/null || {
        echo "  Penn-Fudan下载失败，跳过..."
    }

    if [ -f "PennFudanPed.zip" ]; then
        unzip -q PennFudanPed.zip
        touch .downloaded
        echo "  Penn-Fudan下载完成"
    fi
fi

cd ..

# 创建数据集配置文件
echo ""
echo "创建数据集配置..."

cat > widerperson.yaml << 'EOF'
# WiderPerson Dataset for YOLO
# 专门为行人检测设计

path: /root/pedestrian_training/datasets/widerperson
train: train/images
val: val/images

# 只检测行人
names:
  0: pedestrian

# 数据增强 - 针对行人检测优化
# mosaic: 1.0
# mixup: 0.15
# copy_paste: 0.1
EOF

echo ""
echo "========================================="
echo "配置文件: widerperson.yaml"
echo ""
echo "如果自动下载失败，请手动下载后解压到:"
echo "  ~/pedestrian_training/datasets/widerperson/"
echo ""
echo "目录结构应为:"
echo "  widerperson/"
echo "    train/"
echo "      images/"
echo "      labels/"
echo "    val/"
echo "      images/"
echo "      labels/"
echo "========================================="
