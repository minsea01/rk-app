#!/bin/bash
# CityPersons 数据集下载脚本
# 数据集来源: Cityscapes 官方 + CityPersons 标注

set -e

cd ~/pedestrian_training/datasets

echo "========================================="
echo "CityPersons 数据集下载"
echo "========================================="

# 方案1: 使用预处理好的YOLO格式数据集 (推荐，更快)
echo "[方案1] 从Roboflow下载YOLO格式CityPersons..."
echo "访问: https://universe.roboflow.com/search?q=citypersons"
echo ""

# 方案2: 手动下载 (需要Cityscapes账号)
echo "[方案2] 手动下载步骤:"
echo "1. 注册 https://www.cityscapes-dataset.com/"
echo "2. 下载 leftImg8bit_trainvaltest.zip (11GB)"
echo "3. 下载 gtBbox_cityPersons_trainval.zip (标注)"
echo ""

# 方案3: 使用镜像源 (如果有)
mkdir -p citypersons/{train,val}/{images,labels}

# 创建数据集配置文件
cat > citypersons.yaml << 'EOF'
# CityPersons Dataset for YOLO
# 下载后修改path为实际路径

path: /root/pedestrian_training/datasets/citypersons
train: train/images
val: val/images

# 只检测行人
names:
  0: pedestrian

# 数据增强建议
# hsv_h: 0.015
# hsv_s: 0.7
# hsv_v: 0.4
# degrees: 0.0
# translate: 0.1
# scale: 0.5
# mosaic: 1.0
# mixup: 0.1
EOF

echo ""
echo "========================================="
echo "数据集配置文件已创建: citypersons.yaml"
echo ""
echo "快速方案 (使用COCO person子集测试流程):"
echo "  python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt').train(data='coco8.yaml', epochs=1)\""
echo ""
echo "如需使用CityPersons，请按上述方案下载数据集"
echo "========================================="
