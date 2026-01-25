#!/bin/bash
# CrowdHuman 数据集下载脚本
# 大型行人检测数据集 - 15000训练 + 4370验证
# 包含拥挤场景，非常适合工业应用

set -e

cd ~/pedestrian_training
mkdir -p datasets/crowdhuman
cd datasets/crowdhuman

echo "========================================="
echo "CrowdHuman 数据集下载"
echo "大型行人检测数据集 (约20GB)"
echo "========================================="

# CrowdHuman官方地址（Google Drive，中国可能无法访问）
echo ""
echo "CrowdHuman 是目前最大的行人检测数据集之一"
echo "  训练集: 15000 图像"
echo "  验证集: 4370 图像"
echo "  标注: Full body + Visible body + Head"
echo ""

# 方案1: 从百度网盘镜像下载
echo "[方案1] 百度网盘镜像 (推荐):"
echo "  搜索: CrowdHuman YOLO格式"
echo "  或访问: https://github.com/Megvii-BaseDetection/CrowdHuman"
echo ""

# 方案2: 使用gdown从Google Drive下载
echo "[方案2] 使用gdown下载 (需要VPN)..."

pip install gdown -q

# 创建目录结构
mkdir -p {train,val}/{images,labels}

# 尝试下载
python3 << 'PYEOF'
import os
import subprocess

print("尝试下载CrowdHuman...")

# CrowdHuman YOLO格式 (来自社区转换)
# 尝试几个可能的源

sources = [
    # 可能的Kaggle数据集
    "kaggle datasets download -d crowdhuman/crowdhuman-full-body",
    # 可能的huggingface源
]

print("CrowdHuman数据集较大(~20GB)，建议手动下载")
print("")
print("YOLO格式转换脚本:")
print("  git clone https://github.com/xingyizhou/CrowdHuman")
print("  python convert_crowdhuman_to_yolo.py")

PYEOF

# 方案3: 从Roboflow下载替代数据集
echo ""
echo "[方案3] Roboflow人群检测数据集..."

pip install roboflow -q

python3 << 'PYEOF'
try:
    from roboflow import Roboflow
    import os

    print("搜索Roboflow行人检测数据集...")

    # 几个高质量的行人/人群检测数据集
    datasets_to_try = [
        ("roboflow-universe-projects", "pedestrian-detection-12keo", 1),
        ("people-detection-3", "people-detection-general", 2),
    ]

    for workspace, project_name, version in datasets_to_try:
        try:
            rf = Roboflow()
            project = rf.workspace(workspace).project(project_name)
            dataset = project.version(version).download("yolov8")
            print(f"✅ 成功下载: {project_name}")
            break
        except Exception as e:
            print(f"  {project_name}: {e}")
            continue

except ImportError:
    print("roboflow未安装，请运行: pip install roboflow")
except Exception as e:
    print(f"下载失败: {e}")

PYEOF

# 创建数据集配置文件
cat > crowdhuman.yaml << 'EOF'
# CrowdHuman Dataset for YOLO
# 大型行人/人群检测数据集

path: /root/pedestrian_training/datasets/crowdhuman
train: train/images
val: val/images

# 类别定义
names:
  0: person

# 针对拥挤场景优化的增强参数
# mosaic: 1.0
# mixup: 0.2
# copy_paste: 0.15
EOF

echo ""
echo "========================================="
echo "配置文件已创建: crowdhuman.yaml"
echo ""
echo "推荐下载方式:"
echo "1. 百度网盘搜索 'CrowdHuman YOLO'"
echo "2. Kaggle搜索 'CrowdHuman'"
echo "3. GitHub仓库自行转换"
echo ""
echo "下载后解压到:"
echo "  ~/pedestrian_training/datasets/crowdhuman/"
echo "========================================="
