#!/bin/bash
# 精度修复专用训练脚本
# 解决类别不平衡 + 小目标问题

yolo train \
    data=$HOME/datasets/industrial_15_classes_ready/data.yaml \
    model=yolov8s.pt \
    imgsz=960 \
    epochs=150 \
    batch=auto \
    device=0 \
    mosaic=1.0 \
    mixup=0.1 \
    copy_paste=0.1 \
    fl_gamma=1.5 \
    cos_lr=True \
    lr0=0.005 \
    lrf=0.1 \
    warmup_epochs=5 \
    multi_scale=True \
    cache=ram \
    patience=80 \
    name=precision_fix_final \
    project=runs/train