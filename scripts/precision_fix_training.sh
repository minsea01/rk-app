#!/bin/bash
# 精度修复专用训练脚本
# 解决类别不平衡 + 小目标问题

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$PROJECT_ROOT/scripts/lib/deprecation.sh"
warn_deprecated "scripts/precision_fix_training.sh" "scripts/train.sh"

bash "$PROJECT_ROOT/scripts/train.sh" \
    --profile none \
    --data "$HOME/datasets/industrial_15_classes_ready/data.yaml" \
    --model yolov8s.pt \
    --imgsz 960 \
    --epochs 150 \
    --batch auto \
    --device 0 \
    --cache ram \
    --patience 80 \
    --name precision_fix_final \
    --project runs/train \
    --lr0 0.005 \
    --lrf 0.1 \
    --warmup-epochs 5 \
    --extra mosaic=1.0 \
    --extra mixup=0.1 \
    --extra copy_paste=0.1 \
    --extra fl_gamma=1.5 \
    --extra cos_lr=True \
    --extra multi_scale=True \
    --no-export
