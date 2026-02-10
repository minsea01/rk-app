#!/bin/bash
set -euo pipefail

echo "🏭 MVTec AD工业检测训练方案"
echo "基于你之前成功的训练经验优化"
echo "=" * 50

# 设置保守的CUDA环境（沿用你成功的配置）
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_DATALOADER_PIN_MEMORY=False

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$PROJECT_ROOT/scripts/lib/deprecation.sh"
warn_deprecated "scripts/train_mvtec_industrial.sh" "scripts/train.sh"

echo "🚀 开始MVTec AD工业15类检测训练..."
echo "使用与person detection相同的稳定参数"

# 第一阶段：保守训练（确保稳定）
echo ""
echo "📍 第一阶段：稳定性测试训练（10轮）"
TEST_NAME="mvtec_test_$(date +%m%d_%H%M)"
bash "$PROJECT_ROOT/scripts/train.sh" \
  --profile none \
  --data "$HOME/datasets/mvtec_industrial/data.yaml" \
  --model yolo11s.pt \
  --epochs 10 \
  --imgsz 640 \
  --device 0 \
  --batch 8 \
  --workers 0 \
  --cache False \
  --save-period 5 \
  --project runs/detect \
  --name "$TEST_NAME" \
  --no-export

echo "如果10轮训练稳定，继续完整训练..."

# 第二阶段：完整训练
echo ""  
echo "📍 第二阶段：完整训练（100轮）"
FULL_NAME="mvtec_industrial_$(date +%m%d_%H%M)"
bash "$PROJECT_ROOT/scripts/train.sh" \
  --profile none \
  --data "$HOME/datasets/mvtec_industrial/data.yaml" \
  --model yolo11s.pt \
  --epochs 100 \
  --imgsz 640 \
  --device 0 \
  --batch 8 \
  --workers 0 \
  --cache False \
  --save-period 10 \
  --patience 20 \
  --project runs/detect \
  --name "$FULL_NAME" \
  --no-export

echo ""
echo "🎯 训练完成后自动测试："
echo "yolo detect val model=runs/detect/mvtec_industrial_*/weights/best.pt data=$HOME/datasets/mvtec_industrial/data.yaml"

echo ""
echo "🔄 为RK3588准备ONNX导出："
echo "yolo detect export model=runs/detect/mvtec_industrial_*/weights/best.pt format=onnx opset=12 simplify=True"
