#!/bin/bash
# YOLOv8行人检测模型训练脚本
# 目标：mAP@0.5 > 90%，模型<5MB

set -e

# ============ 配置参数 ============
DATA_YAML="${1:-datasets/coco_person/data.yaml}"
MODEL="${2:-yolov8s.pt}"  # 使用YOLOv8s达到>90% mAP
IMGSZ=640                  # 训练分辨率（高精度）
EXPORT_IMGSZ=416          # 导出分辨率（NPU优化）
EPOCHS=100
BATCH_SIZE=16
DEVICE=0
PROJECT="runs/train_pedestrian"
NAME="pedestrian_yolov8s"

echo "============================================"
echo "行人检测模型训练"
echo "============================================"
echo "数据集: $DATA_YAML"
echo "基础模型: $MODEL"
echo "训练分辨率: ${IMGSZ}x${IMGSZ}"
echo "导出分辨率: ${EXPORT_IMGSZ}x${EXPORT_IMGSZ}"
echo "训练轮数: $EPOCHS"
echo "============================================"

# 激活环境
source ~/yolo_env/bin/activate

# ============ 阶段1：模型训练 ============
echo ""
echo "[1/5] 开始训练..."
python -c "
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('$MODEL')

# 训练配置
results = model.train(
    data='$DATA_YAML',
    epochs=$EPOCHS,
    imgsz=$IMGSZ,
    batch=$BATCH_SIZE,
    device=$DEVICE,
    project='$PROJECT',
    name='$NAME',

    # 优化超参数（针对行人检测）
    patience=20,              # Early stopping
    save=True,
    save_period=10,

    # 数据增强
    mosaic=1.0,              # Mosaic增强
    mixup=0.1,               # Mixup增强
    copy_paste=0.1,          # 复制粘贴增强
    degrees=10.0,            # 旋转角度
    translate=0.1,           # 平移
    scale=0.5,               # 缩放
    shear=5.0,               # 错切
    flipud=0.0,              # 不垂直翻转（行人通常竖直）
    fliplr=0.5,              # 水平翻转

    # 优化器
    optimizer='AdamW',
    lr0=0.001,               # 初始学习率
    lrf=0.01,                # 最终学习率比例
    momentum=0.937,
    weight_decay=0.0005,

    # 其他
    workers=8,
    verbose=True,
)

print('\n训练完成！')
print(f'最佳权重: {model.trainer.best}')
print(f'结果保存至: {model.trainer.save_dir}')
"

# ============ 阶段2：模型验证 ============
BEST_WEIGHT="$PROJECT/$NAME/weights/best.pt"

if [ ! -f "$BEST_WEIGHT" ]; then
    echo "错误：未找到最佳权重文件 $BEST_WEIGHT"
    exit 1
fi

echo ""
echo "[2/5] 验证模型精度..."
python -c "
from ultralytics import YOLO

model = YOLO('$BEST_WEIGHT')
metrics = model.val(
    data='$DATA_YAML',
    imgsz=$IMGSZ,
    batch=$BATCH_SIZE,
    device=$DEVICE,
)

# 打印关键指标
print('\n========== 验证结果 ==========')
print(f'mAP@0.5: {metrics.box.map50:.4f}')
print(f'mAP@0.5:0.95: {metrics.box.map:.4f}')
print(f'Precision: {metrics.box.mp:.4f}')
print(f'Recall: {metrics.box.mr:.4f}')

# 检查是否满足要求
if metrics.box.map50 > 0.90:
    print('\n✓ 精度达标！(mAP@0.5 > 90%)')
else:
    print(f'\n⚠ 精度未达标：{metrics.box.map50:.2%} < 90%')
    print('建议：')
    print('  1. 增加训练轮数 (--epochs)')
    print('  2. 使用更大模型 (yolov8m.pt)')
    print('  3. 调整数据增强参数')
"

# ============ 阶段3：导出ONNX（640分辨率）============
echo ""
echo "[3/5] 导出ONNX模型（640分辨率）..."
python tools/export_yolov8_to_onnx.py \
  --weights "$BEST_WEIGHT" \
  --imgsz 640 \
  --opset 12 \
  --simplify \
  --outdir artifacts/models \
  --outfile pedestrian_640.onnx

# ============ 阶段4：导出ONNX（416分辨率，NPU优化）============
echo ""
echo "[4/5] 导出ONNX模型（416分辨率，NPU优化）..."
python tools/export_yolov8_to_onnx.py \
  --weights "$BEST_WEIGHT" \
  --imgsz $EXPORT_IMGSZ \
  --opset 12 \
  --simplify \
  --outdir artifacts/models \
  --outfile pedestrian_416.onnx

# ============ 阶段5：转换RKNN并量化 ============
echo ""
echo "[5/5] 转换为RKNN并进行INT8量化..."

# 生成校准集（从训练集抽取）
python tools/make_calib_set.py \
  --data "$DATA_YAML" \
  --output datasets/pedestrian_calib \
  --num 300

# 转换为RKNN
python tools/convert_onnx_to_rknn.py \
  --onnx artifacts/models/pedestrian_416.onnx \
  --out artifacts/models/pedestrian_416.rknn \
  --calib datasets/pedestrian_calib/calib.txt \
  --target rk3588 \
  --do-quant

# 检查模型大小
MODEL_SIZE=$(du -h artifacts/models/pedestrian_416.rknn | cut -f1)
echo ""
echo "========== 模型信息 =========="
echo "RKNN模型大小: $MODEL_SIZE"

# ============ 完成 ============
echo ""
echo "============================================"
echo "✓ 训练流程完成！"
echo "============================================"
echo ""
echo "生成的文件："
echo "  训练权重: $BEST_WEIGHT"
echo "  ONNX模型: artifacts/models/pedestrian_640.onnx"
echo "  ONNX模型(NPU): artifacts/models/pedestrian_416.onnx"
echo "  RKNN模型: artifacts/models/pedestrian_416.rknn"
echo ""
echo "下一步："
echo "  1. PC验证: python scripts/run_rknn_sim.py"
echo "  2. 精度对比: python scripts/compare_onnx_rknn.py"
echo "  3. 板端部署: bash scripts/deploy/deploy_to_board.sh"
