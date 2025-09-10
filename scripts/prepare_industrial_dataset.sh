#!/bin/bash

# 工业数据集准备脚本
# 适用于RK3588项目要求（>10类，>24FPS推理）

set -e

DATASET_ROOT="/home/minsea01/datasets"
PROJECT_ROOT="/home/minsea01/dev/rk-projects/rk-app"

echo "🔧 准备工业检测数据集..."

# 1. 创建目标目录
mkdir -p ${DATASET_ROOT}/industrial_detection_v2/{train,val,test}/{images,labels}

# 2. 下载MVTec异常检测数据集
cd ${DATASET_ROOT}
if [ ! -d "mvtec_anomaly_detection" ]; then
    echo "📥 下载MVTec异常检测数据集..."
    python download_mvtec_ad.py --target-dir mvtec_anomaly_detection
fi

# 3. 下载Roboflow工业数据集
if [ ! -d "roboflow_industrial" ]; then
    echo "📥 下载Roboflow工业数据集..."
    ./download_roboflow.py --project "industrial-parts-detection" --version 2
fi

# 4. 合并并平衡数据集
echo "⚖️ 平衡数据集类别分布..."
python ${PROJECT_ROOT}/tools/balance_industrial_dataset.py \
    --input-dirs mvtec_anomaly_detection roboflow_industrial \
    --output-dir industrial_detection_v2 \
    --min-samples-per-class 300 \
    --train-ratio 0.7 \
    --val-ratio 0.2 \
    --test-ratio 0.1

# 5. 生成数据集配置
cat > ${DATASET_ROOT}/industrial_detection_v2/data.yaml << EOF
path: ${DATASET_ROOT}/industrial_detection_v2
train: train/images
val: val/images
test: test/images

nc: 15
names:
  0: screw
  1: bolt  
  2: nut
  3: washer
  4: gear
  5: bearing
  6: circuit_board
  7: connector
  8: sensor
  9: cable
  10: valve
  11: pump
  12: motor
  13: pipe
  14: defect
EOF

echo "✅ 工业数据集准备完成！"
echo "📊 数据集位置: ${DATASET_ROOT}/industrial_detection_v2/"
echo "🎯 类别数量: 15"
echo "📈 推荐训练配置: YOLOv8s, 640px, 100 epochs"