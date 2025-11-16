# 自定义数据集训练与部署指南

本文档详细说明如何使用自己的数据集训练 YOLO 模型，并部署到 RK3588 平台。

## 目录

1. [数据集准备](#1-数据集准备)
2. [模型训练](#2-模型训练)
3. [模型导出（ONNX）](#3-模型导出onnx)
4. [校准数据集准备](#4-校准数据集准备)
5. [RKNN 转换](#5-rknn-转换)
6. [配置文件更新](#6-配置文件更新)
7. [验证测试](#7-验证测试)
8. [性能优化](#8-性能优化)

---

## 1. 数据集准备

### 1.1 YOLO 格式数据集结构

```
your_dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   ├── val/
│   │   ├── img003.jpg
│   │   └── img004.jpg
│   └── test/           # 可选
│       └── img005.jpg
├── labels/
│   ├── train/
│   │   ├── img001.txt
│   │   └── img002.txt
│   └── val/
│       ├── img003.txt
│       └── img004.txt
└── data.yaml
```

### 1.2 标注文件格式（.txt）

每个图像对应一个 `.txt` 文件，每行表示一个目标：

```
<class_id> <x_center> <y_center> <width> <height>
```

- 坐标归一化到 [0, 1]
- 示例：`0 0.5 0.5 0.3 0.4`（类别0，中心点0.5,0.5，宽0.3，高0.4）

### 1.3 数据配置文件（data.yaml）

```yaml
# your_dataset/data.yaml
path: /home/minsea/rk-app/datasets/your_dataset  # 数据集根目录（绝对路径）
train: images/train  # 训练集（相对于 path）
val: images/val      # 验证集
test: images/test    # 测试集（可选）

# 类别定义
nc: 2  # 类别数量
names: ['person', 'vehicle']  # 类别名称列表（索引对应 class_id）
```

### 1.4 数据集转换工具

如果你的数据集是其他格式（COCO、VOC、LabelMe 等），可以使用转换脚本：

```bash
# COCO → YOLO
python tools/convert_coco_to_yolo.py \
  --coco-json datasets/coco/annotations/instances_train2017.json \
  --images-dir datasets/coco/train2017 \
  --output datasets/your_dataset

# VOC → YOLO
python tools/convert_voc_to_yolo.py \
  --voc-dir datasets/VOCdevkit/VOC2012 \
  --output datasets/your_dataset
```

**注意**：当前项目没有提供转换脚本，需要自行实现或使用第三方工具（如 `roboflow`、`labelImg` 导出）。

---

## 2. 模型训练

### 2.1 激活虚拟环境

```bash
source ~/yolo_env/bin/activate
export PYTHONPATH=/home/minsea/rk-app
```

### 2.2 训练命令（Ultralytics CLI）

```bash
# 基础训练（从预训练模型 fine-tune）
yolo detect train \
  model=yolov8n.pt \
  data=/home/minsea/rk-app/datasets/your_dataset/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=runs/train \
  name=your_model

# 从头训练（不推荐，除非数据量 >10k）
yolo detect train \
  model=yolov8n.yaml \
  data=datasets/your_dataset/data.yaml \
  epochs=300 \
  imgsz=640 \
  pretrained=False
```

### 2.3 训练参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `model` | 基础模型（.pt）或配置（.yaml） | yolov8n.pt（轻量）<br>yolov8s.pt（平衡） |
| `epochs` | 训练轮数 | 100-300（根据数据量） |
| `imgsz` | 输入尺寸 | 640（标准）<br>416（高速，见下文优化） |
| `batch` | 批大小 | 16-32（GPU 内存允许） |
| `device` | GPU 设备 | 0（单卡）或 0,1（多卡） |
| `patience` | 早停轮数 | 50（默认） |
| `optimizer` | 优化器 | SGD（默认）或 Adam |

### 2.4 训练输出

训练完成后，最佳模型保存在：

```
runs/train/your_model/
├── weights/
│   ├── best.pt       # 验证集最佳模型（用于部署）
│   └── last.pt       # 最后一轮模型
├── results.png       # 训练曲线
└── confusion_matrix.png  # 混淆矩阵
```

### 2.5 训练监控

```bash
# 实时查看训练日志
tail -f runs/train/your_model/results.csv

# 使用 TensorBoard（如果启用）
tensorboard --logdir runs/train
```

---

## 3. 模型导出（ONNX）

### 3.1 使用项目提供的导出脚本

```bash
python3 tools/export_yolov8_to_onnx.py \
  --weights runs/train/your_model/weights/best.pt \
  --imgsz 640 \
  --outdir artifacts/models \
  --simplify
```

**输出**：`artifacts/models/best.onnx`

### 3.2 使用 Ultralytics CLI 导出

```bash
yolo export \
  model=runs/train/your_model/weights/best.pt \
  format=onnx \
  imgsz=640 \
  opset=12 \
  simplify=True
```

### 3.3 验证 ONNX 模型

```bash
# PC GPU 推理测试（确保 ONNX 正确）
yolo predict \
  model=artifacts/models/best.onnx \
  source=assets/test.jpg \
  imgsz=640 \
  conf=0.5 \
  save=true
```

如果推理失败或结果异常，**不要**继续 RKNN 转换，先修复 ONNX 模型。

---

## 4. 校准数据集准备

INT8 量化需要校准数据集（通常 100-500 张图像），用于统计激活值分布。

### 4.1 从训练集选择代表性图像

```bash
# 创建校准数据集目录
mkdir -p datasets/your_dataset/calib_images

# 随机选取 300 张训练集图像
cd datasets/your_dataset
find images/train -name "*.jpg" | shuf -n 300 | xargs -I {} cp {} calib_images/

# 生成绝对路径列表（必须使用绝对路径！）
find calib_images -name "*.jpg" -exec realpath {} \; > calib_images/calib.txt
```

### 4.2 校准数据集要求

- **数量**：100-500 张（推荐 300 张）
- **多样性**：覆盖不同光照、角度、遮挡情况
- **格式**：与训练集相同（JPG/PNG）
- **路径**：必须使用绝对路径（`realpath` 或 `pwd`）

**⚠️ 常见错误**：使用相对路径会导致转换时路径重复，如：
```
/home/user/rk-app/datasets/coco/calib_images/datasets/coco/calib_images/img.jpg
```

---

## 5. RKNN 转换

### 5.1 基础转换（INT8 量化）

```bash
python3 tools/convert_onnx_to_rknn.py \
  --onnx artifacts/models/best.onnx \
  --out artifacts/models/your_model_int8.rknn \
  --calib datasets/your_dataset/calib_images/calib.txt \
  --target rk3588 \
  --do-quant
```

### 5.2 高级参数

```bash
python3 tools/convert_onnx_to_rknn.py \
  --onnx artifacts/models/best.onnx \
  --out artifacts/models/your_model_416_int8.rknn \
  --calib datasets/your_dataset/calib_images/calib.txt \
  --target rk3588 \
  --do-quant \
  --size 416                    # 推荐：避免 Transpose CPU 回退
  # --quant-dtype w8a8           # 可选：显式指定量化类型
  # --optimization-level 3       # 可选：优化等级（1-3）
```

### 5.3 转换输出

```
Creating quantized model...
Model size: 4.7 MB
Done! Output: artifacts/models/your_model_int8.rknn
```

---

## 6. 配置文件更新

### 6.1 更新检测配置（config/detection/detect.yaml）

```yaml
# config/detection/detect_your_model.yaml
model:
  path: artifacts/models/your_model_int8.rknn
  size: 416               # 与转换时的 --size 一致
  conf_threshold: 0.5     # 置信度阈值（根据数据集调整）
  iou_threshold: 0.45     # NMS IoU 阈值
  max_detections: 3549    # 416×416 对应的最大检测数

classes:
  num: 2                  # 类别数量（与 data.yaml 的 nc 一致）
  names:
    - person
    - vehicle

input:
  format: NHWC            # RKNN 输入格式（不变）
  dtype: uint8            # RKNN 输入类型（不变）
  mean: [0, 0, 0]         # 预处理均值（通常不需要修改）
  std: [255, 255, 255]    # 预处理标准差
```

### 6.2 更新 apps/config.py（如果需要）

如果你的模型有特殊输出格式（如自定义 head），可能需要修改 [apps/config.py](apps/config.py)：

```python
class ModelConfig:
    DEFAULT_SIZE = 416  # 修改默认尺寸

    # 如果类别数变化，更新这里
    NUM_CLASSES = 2  # 你的数据集类别数
```

---

## 7. 验证测试

### 7.1 PC RKNN 模拟器验证

```bash
# 使用项目提供的模拟器脚本
python scripts/run_rknn_sim.py \
  --model artifacts/models/your_model_int8.rknn \
  --image assets/test.jpg \
  --size 416
```

### 7.2 ONNX vs RKNN 精度对比

```bash
python scripts/compare_onnx_rknn.py \
  --onnx artifacts/models/best.onnx \
  --rknn artifacts/models/your_model_int8.rknn \
  --images datasets/your_dataset/calib_images \
  --size 416
```

**期望结果**：
- 平均绝对误差 <1%
- 最大相对误差 <5%

### 7.3 在 RK3588 板上测试（需要硬件）

```bash
# 部署到板子
scripts/deploy/rk3588_run.sh \
  --model artifacts/models/your_model_int8.rknn \
  --config config/detection/detect_your_model.yaml

# 或使用 Python runner
scripts/deploy/rk3588_run.sh \
  --runner python \
  --model artifacts/models/your_model_int8.rknn
```

---

## 8. 性能优化

### 8.1 输入尺寸选择

| 尺寸 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 416×416 | **全 NPU 执行**（无 CPU 回退）<br>推理速度快 | 精度略低 | 工业实时应用（>30 FPS） |
| 640×640 | 精度最高 | Transpose 层 CPU 回退<br>速度略慢 | 精度敏感场景 |
| 320×320 | 速度最快 | 精度最低 | 低算力设备或超高帧率需求 |

**推荐**：优先使用 416×416，除非精度无法满足需求。

### 8.2 置信度阈值调优

```bash
# 测试不同阈值对 FPS 的影响
for conf in 0.25 0.35 0.5 0.6; do
  echo "Testing conf=$conf"
  python apps/yolov8_rknn_infer.py \
    --conf $conf \
    --model artifacts/models/your_model_int8.rknn \
    --image assets/test.jpg
done
```

**关键发现**（参考 CLAUDE.md）：
- conf=0.25（默认）：后处理 3135ms → 0.3 FPS（NMS 瓶颈）
- conf=0.5（优化）：后处理 5.2ms → 60+ FPS

**建议**：工业应用使用 conf≥0.5，避免过多误检导致 NMS 开销。

### 8.3 数据集质量优化

- **标注质量**：检查并修正标注错误（使用 [labelImg](https://github.com/heartexlabs/labelImg) 或 [CVAT](https://github.com/opencv/cvat)）
- **数据增强**：训练时启用 Mosaic、MixUp、HSV 增强（Ultralytics 默认开启）
- **困难样本挖掘**：将推理失败的案例加入训练集

### 8.4 模型剪枝（可选）

如果模型仍然过大或速度不够：

```bash
# 使用 YOLOv8 Nano（最小模型）
yolo detect train model=yolov8n.pt ...

# 或使用模型剪枝工具（需额外安装）
# pip install torch-pruning
python tools/prune_model.py \
  --model runs/train/your_model/weights/best.pt \
  --pruning-ratio 0.3
```

---

## 9. 完整工作流示例

假设你有一个行人检测数据集 `pedestrian_dataset/`：

```bash
# 1. 激活环境
source ~/yolo_env/bin/activate
export PYTHONPATH=/home/minsea/rk-app

# 2. 训练模型
yolo detect train \
  model=yolov8n.pt \
  data=datasets/pedestrian_dataset/data.yaml \
  epochs=100 \
  imgsz=416 \
  batch=16 \
  device=0 \
  project=runs/train \
  name=pedestrian_detector

# 3. 导出 ONNX
python3 tools/export_yolov8_to_onnx.py \
  --weights runs/train/pedestrian_detector/weights/best.pt \
  --imgsz 416 \
  --outdir artifacts/models

# 4. 准备校准数据集
mkdir -p datasets/pedestrian_dataset/calib_images
find datasets/pedestrian_dataset/images/train -name "*.jpg" | shuf -n 300 | \
  xargs -I {} cp {} datasets/pedestrian_dataset/calib_images/
find datasets/pedestrian_dataset/calib_images -name "*.jpg" -exec realpath {} \; > \
  datasets/pedestrian_dataset/calib_images/calib.txt

# 5. 转换 RKNN
python3 tools/convert_onnx_to_rknn.py \
  --onnx artifacts/models/best.onnx \
  --out artifacts/models/pedestrian_int8_416.rknn \
  --calib datasets/pedestrian_dataset/calib_images/calib.txt \
  --target rk3588 \
  --do-quant \
  --size 416

# 6. PC 模拟器验证
python scripts/run_rknn_sim.py \
  --model artifacts/models/pedestrian_int8_416.rknn \
  --image assets/test_pedestrian.jpg \
  --size 416

# 7. 板上部署（需硬件）
scripts/deploy/rk3588_run.sh \
  --model artifacts/models/pedestrian_int8_416.rknn \
  --config config/detection/detect_pedestrian.yaml
```

---

## 10. 常见问题

### Q1: 训练后 mAP 很低怎么办？

**A**: 检查以下方面：
1. 标注质量（使用验证工具检查）
2. 数据分布（训练集和验证集是否平衡）
3. 训练轮数（是否收敛，查看 `results.png`）
4. 数据增强（尝试调整 `augment` 参数）
5. 学习率（默认 0.01，可尝试 0.001-0.1）

### Q2: RKNN 转换后精度大幅下降？

**A**:
1. 检查校准数据集质量（是否代表真实分布）
2. 增加校准图像数量（300 → 500）
3. 尝试不同量化类型（`--quant-dtype w8a16`）
4. 对比 ONNX vs RKNN 输出（`compare_onnx_rknn.py`）

### Q3: 板上推理速度达不到 30 FPS？

**A**:
1. 使用 416×416 分辨率（避免 Transpose CPU 回退）
2. 提高置信度阈值（conf≥0.5）
3. 启用 NPU 多核并行（设置 `core_mask`）
4. 检查是否有层回退到 CPU（使用 `rknn.eval_perf()`）

### Q4: 如何评估模型在自定义数据集上的 mAP？

**A**: 使用 Ultralytics 提供的验证工具：

```bash
yolo detect val \
  model=runs/train/your_model/weights/best.pt \
  data=datasets/your_dataset/data.yaml \
  imgsz=416
```

或使用项目脚本（如果提供）：
```bash
python scripts/evaluate_map.py \
  --model artifacts/models/your_model_int8.rknn \
  --data datasets/your_dataset/data.yaml
```

---

## 11. 相关文档

- [CLAUDE.md](../CLAUDE.md) - 项目总览与架构
- [PERSON_TRAINING_GUIDE.md](PERSON_TRAINING_GUIDE.md) - 行人检测专用训练指南
- [Ultralytics 官方文档](https://docs.ultralytics.com/)
- [RKNN-Toolkit2 文档](https://github.com/rockchip-linux/rknn-toolkit2)

---

**最后更新**：2025-10-30
**作者**：Claude Code
**版本**：v1.0
