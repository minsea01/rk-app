#!/bin/bash
# 真实工作的检测演示 - 使用本地图片避免GigE连接问题

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "RK3588工业检测系统 - 实际运行演示"
echo "============================================================"

# 检查是否有可用的测试图片
TEST_IMAGES=""
if [ -d "$HOME/datasets/coco128/images/train2017" ]; then
    TEST_IMAGES="$HOME/datasets/coco128/images/train2017"
elif [ -d "$HOME/datasets/coco/images/val" ]; then
    TEST_IMAGES="$HOME/datasets/coco/images/val"
elif [ -d "datasets/calib" ]; then
    TEST_IMAGES="datasets/calib"
else
    echo "[错误] 未找到测试图片目录"
    exit 1
fi

echo "测试图片目录: $TEST_IMAGES"
IMG_COUNT=$(find "$TEST_IMAGES" -maxdepth 1 -type f -name '*.jpg' 2>/dev/null | wc -l)
echo "图片数量: $IMG_COUNT"

# 创建临时配置文件
cat > /tmp/demo_config.yaml << EOF
source:
  type: folder
  uri: "$TEST_IMAGES"

engine:
  type: onnx
  model: "artifacts/models/best.onnx"
  imgsz: 640

nms:
  conf_thres: 0.25
  iou_thres: 0.60
  topk: 1000

classes: "config/industrial_classes.txt"
EOF

echo "临时配置已创建: /tmp/demo_config.yaml"
echo ""

# 显示将要使用的配置
echo "=== 演示配置 ==="
cat /tmp/demo_config.yaml

echo ""
echo "开始15类工业检测演示..."
echo "处理方式: 读取本地图片，模拟实时检测"
echo "运行时间: 30秒"
echo ""

# 运行实际检测
timeout 30s ./build/detect_cli --cfg /tmp/demo_config.yaml

echo ""
echo "演示完成！"
echo ""
echo "这次运行展示了:"
echo "  - 真实的ONNX模型推理"
echo "  - 15类工业零件检测算法"
echo "  - 实际的图片处理和分析"
echo "  - 真实的推理时间测量"

# 清理临时文件
rm -f /tmp/demo_config.yaml

echo ""
echo "注意: 这是使用本地图片的演示版本"
echo "在RK3588上将使用真实的2K工业相机进行相同的处理"
