#!/bin/bash
# 全自动行人检测模型训练流程
# 功能：自动下载数据、训练、导出、转换RKNN
# 可后台运行，支持断点续传

set -euo pipefail

# ============ 配置参数 ============
LOG_FILE="logs/auto_train_$(date +%Y%m%d_%H%M%S).log"
COCO_DIR="datasets/coco_raw"
DATASET_DIR="datasets/coco_person"
MODEL="yolov8s.pt"
MIN_DISK_SPACE_GB=30  # 最小磁盘空间要求（GB）

# 创建日志目录
mkdir -p logs

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$LOG_FILE" >&2
}

# ============ 前置检查 ============
log "=========================================="
log "自动训练流程启动"
log "=========================================="

# 检查磁盘空间
log "[1/10] 检查磁盘空间..."
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt "$MIN_DISK_SPACE_GB" ]; then
    log_error "磁盘空间不足！需要至少 ${MIN_DISK_SPACE_GB}GB，当前可用 ${AVAILABLE_SPACE}GB"
    exit 1
fi
log "✓ 磁盘空间充足: ${AVAILABLE_SPACE}GB 可用"

# 检查GPU
log "[2/10] 检查GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    log "✓ 检测到GPU: $GPU_INFO"
else
    log "⚠ 未检测到GPU，训练将使用CPU（速度较慢）"
fi

# 激活Python环境
log "[3/10] 激活Python环境..."
source ~/yolo_env/bin/activate || {
    log_error "无法激活yolo_env环境"
    exit 1
}
log "✓ Python环境已激活: $(python --version)"

# ============ 下载COCO数据集 ============
log "[4/10] 检查COCO数据集..."

if [ -f "$COCO_DIR/annotations/instances_train2017.json" ]; then
    log "✓ COCO数据集已存在，跳过下载"
else
    log "开始下载COCO 2017数据集（约20GB，可能需要较长时间）..."
    
    mkdir -p "$COCO_DIR"
    cd "$COCO_DIR"
    
    # 下载训练集图像
    if [ ! -f "train2017.zip" ]; then
        log "  下载训练集图像..."
        wget --continue --progress=bar:force \
          http://images.cocodataset.org/zips/train2017.zip 2>&1 | \
          while read line; do log "$line"; done
    fi
    
    # 下载验证集图像
    if [ ! -f "val2017.zip" ]; then
        log "  下载验证集图像..."
        wget --continue --progress=bar:force \
          http://images.cocodataset.org/zips/val2017.zip 2>&1 | \
          while read line; do log "$line"; done
    fi
    
    # 下载标注文件
    if [ ! -f "annotations_trainval2017.zip" ]; then
        log "  下载标注文件..."
        wget --continue --progress=bar:force \
          http://images.cocodataset.org/annotations/annotations_trainval2017.zip 2>&1 | \
          while read line; do log "$line"; done
    fi
    
    # 解压文件
    log "  解压文件..."
    [ ! -d "train2017" ] && unzip -q train2017.zip
    [ ! -d "val2017" ] && unzip -q val2017.zip
    [ ! -d "annotations" ] && unzip -q annotations_trainval2017.zip
    
    cd - > /dev/null
    log "✓ COCO数据集下载完成"
fi

# ============ 处理Person数据集 ============
log "[5/10] 处理COCO Person数据集..."

if [ -f "$DATASET_DIR/data.yaml" ]; then
    log "✓ Person数据集已处理，跳过"
else
    log "开始处理Person子集..."
    python tools/prepare_coco_person.py \
        --coco-dir "$COCO_DIR" \
        --output-dir "$DATASET_DIR" \
        --copy-images 2>&1 | \
        while read line; do log "$line"; done
    log "✓ Person数据集处理完成"
fi

# ============ 开始训练 ============
log "[6/10] 开始训练YOLOv8s模型..."
log "训练参数："
log "  - 模型: $MODEL"
log "  - 数据集: $DATASET_DIR/data.yaml"
log "  - 训练轮数: 100"
log "  - 预计时间: 6-12小时（取决于GPU）"

TRAIN_START=$(date +%s)

python << EOFPYTHON 2>&1 | while read line; do log "$line"; done
from ultralytics import YOLO

# 加载模型
model = YOLO('$MODEL')

# 训练
results = model.train(
    data='$DATASET_DIR/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    project='runs/train_pedestrian',
    name='auto_train',
    patience=20,
    save=True,
    save_period=10,
    
    # 数据增强
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    shear=5.0,
    flipud=0.0,
    fliplr=0.5,
    
    # 优化器
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    
    workers=8,
    verbose=True,
)

print('\n训练完成！')
print(f'最佳权重: {model.trainer.best}')
EOFPYTHON

TRAIN_END=$(date +%s)
TRAIN_DURATION=$((TRAIN_END - TRAIN_START))
log "✓ 训练完成！耗时: $((TRAIN_DURATION / 3600))小时$((TRAIN_DURATION % 3600 / 60))分钟"

# ============ 验证模型 ============
log "[7/10] 验证模型精度..."

BEST_WEIGHT="runs/train_pedestrian/auto_train/weights/best.pt"

if [ ! -f "$BEST_WEIGHT" ]; then
    log_error "未找到训练权重: $BEST_WEIGHT"
    exit 1
fi

python << EOFPYTHON 2>&1 | while read line; do log "$line"; done
from ultralytics import YOLO

model = YOLO('$BEST_WEIGHT')
metrics = model.val(data='$DATASET_DIR/data.yaml', imgsz=640, batch=16, device=0)

print('\n========== 验证结果 ==========')
print(f'mAP@0.5: {metrics.box.map50:.4f}')
print(f'mAP@0.5:0.95: {metrics.box.map:.4f}')
print(f'Precision: {metrics.box.mp:.4f}')
print(f'Recall: {metrics.box.mr:.4f}')

if metrics.box.map50 > 0.90:
    print('\n✓ 精度达标！(mAP@0.5 > 90%)')
else:
    print(f'\n⚠ 精度未达标：{metrics.box.map50:.2%} < 90%')
EOFPYTHON

# ============ 导出ONNX ============
log "[8/10] 导出ONNX模型..."

log "  导出640分辨率版本..."
python tools/export_yolov8_to_onnx.py \
    --weights "$BEST_WEIGHT" \
    --imgsz 640 \
    --opset 12 \
    --simplify \
    --outdir artifacts/models \
    --outfile pedestrian_640.onnx 2>&1 | while read line; do log "$line"; done

log "  导出416分辨率版本（NPU优化）..."
python tools/export_yolov8_to_onnx.py \
    --weights "$BEST_WEIGHT" \
    --imgsz 416 \
    --opset 12 \
    --simplify \
    --outdir artifacts/models \
    --outfile pedestrian_416.onnx 2>&1 | while read line; do log "$line"; done

log "✓ ONNX模型导出完成"

# ============ 生成校准集 ============
log "[9/10] 生成RKNN量化校准集..."

python tools/make_calib_set.py \
    --data "$DATASET_DIR/data.yaml" \
    --output datasets/pedestrian_calib \
    --num 300 2>&1 | while read line; do log "$line"; done

# ============ 转换RKNN ============
log "[10/10] 转换为RKNN并进行INT8量化..."

python tools/convert_onnx_to_rknn.py \
    --onnx artifacts/models/pedestrian_416.onnx \
    --out artifacts/models/pedestrian_416.rknn \
    --calib datasets/pedestrian_calib/calib.txt \
    --target rk3588 \
    --do-quant 2>&1 | while read line; do log "$line"; done

# 检查模型大小
MODEL_SIZE=$(du -h artifacts/models/pedestrian_416.rknn | cut -f1)
log "RKNN模型大小: $MODEL_SIZE"

# ============ 完成总结 ============
TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - $(head -1 "$LOG_FILE" | grep -oP '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' | xargs -I{} date -d "{}" +%s)))

log ""
log "=========================================="
log "✓ 全流程完成！"
log "=========================================="
log ""
log "总耗时: $((TOTAL_DURATION / 3600))小时$((TOTAL_DURATION % 3600 / 60))分钟"
log ""
log "生成的文件："
log "  训练权重: $BEST_WEIGHT"
log "  ONNX模型: artifacts/models/pedestrian_640.onnx"
log "  ONNX模型(NPU): artifacts/models/pedestrian_416.onnx"
log "  RKNN模型: artifacts/models/pedestrian_416.rknn ($MODEL_SIZE)"
log ""
log "日志文件: $LOG_FILE"
log ""
log "下一步："
log "  1. PC验证: python scripts/run_rknn_sim.py"
log "  2. 精度对比: python scripts/compare_onnx_rknn.py"
log "  3. 板端部署: bash scripts/deploy/deploy_to_board.sh"
log ""
log "=========================================="

# 发送完成通知（可选）
echo "训练完成！查看日志: $LOG_FILE"
