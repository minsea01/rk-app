# COCO 行人检测模型训练指南

## 🎯 目标

从 COCO 数据集中提取行人 (person) 子集，训练专门的行人检测模型，达到 mAP@0.5 > 90%。

---

## 📋 完整流程

### 第一步：准备数据集

```bash
# 从 COCO 中提取行人子集
python3 scripts/prepare_person_dataset.py
```

**这一步会：**
- 扫描 COCO train2017 和 val2017
- 提取所有包含 "person" 类别的图像（预期 ~64,000 训练图像，~2,600 验证图像）
- 转换标注为 YOLO 格式
- 生成 `datasets/coco_person/data.yaml` 配置文件

**输出：**
```
datasets/coco_person/
├── images/
│   ├── train/  (~64,000 张图像)
│   └── val/    (~2,600 张图像)
├── labels/
│   ├── train/  (YOLO 格式标注)
│   └── val/
└── data.yaml   (数据集配置)
```

---

### 第二步：选择模型

| 模型 | 参数量 | 速度 | 精度 | 推荐场景 |
|------|--------|------|------|---------|
| **yolo11n** | 2.6M | 最快 | 中 | 实时性优先（已测试 86% mAP） |
| **yolo11s** | 9.4M | 快 | 高 | **推荐** (平衡性能和精度) |
| **yolo11m** | 20.1M | 中 | 最高 | 精度优先 (可能超 90% mAP) |

**建议：** 先用 **yolo11s** 训练，如果 mAP 还不够再试 yolo11m

---

### 第三步：开始训练

```bash
# 训练 yolo11s (推荐)
bash scripts/train_person_detector.sh yolo11s

# 或训练 yolo11m (精度更高但更慢)
bash scripts/train_person_detector.sh yolo11m

# 或继续用 yolo11n (最快但精度较低)
bash scripts/train_person_detector.sh yolo11n
```

**训练参数：**
- 分辨率: 416×416 (适配 RK3588 NPU)
- Epochs: 100
- Batch size: 16
- Patience: 20 (早停)

**预期时间：**
- yolo11s: 2-4 小时 (GPU: RTX 3060)
- yolo11m: 4-6 小时
- yolo11n: 1-2 小时

---

### 第四步：自动执行的后续步骤

训练脚本会**自动完成**：

1. ✅ 训练模型 → `runs/detect/person_yolo11s_416/weights/best.pt`
2. ✅ 验证精度 → 输出 mAP@0.5 和 mAP@0.5:0.95
3. ✅ 导出 ONNX → `runs/detect/person_yolo11s_416/weights/best.onnx`
4. ✅ 转换 RKNN → `artifacts/models/person_yolo11s_416.rknn`

---

## 📊 训练完成后的检查

### 1. 查看训练结果

```bash
# 查看训练曲线（最后一行是最终结果）
tail -20 runs/detect/person_yolo11s_416/results.csv

# 或用 Python 分析
python3 << 'EOF'
import pandas as pd
df = pd.read_csv("runs/detect/person_yolo11s_416/results.csv")
print(df.tail(1)[['metrics/mAP50(B)', 'metrics/mAP50-95(B)']])
EOF
```

### 2. PC ONNX 性能测试

```bash
source ~/yolo_env/bin/activate
python3 << 'EOF'
import time
import cv2
import numpy as np
import onnxruntime as ort

# 加载新模型
session = ort.InferenceSession(
    "runs/detect/person_yolo11s_416/weights/best.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# 准备测试图像
img = cv2.imread("assets/test.jpg")
img = cv2.resize(img, (416, 416))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_data = img[np.newaxis, :, :, :].transpose(0, 3, 1, 2).astype('float32') / 255.0

# 预热
for _ in range(3):
    session.run(None, {"images": input_data})

# 实测
times = []
for _ in range(20):
    start = time.perf_counter()
    session.run(None, {"images": input_data})
    times.append(time.perf_counter() - start)

avg_time = np.mean(times)
print(f"平均延迟: {avg_time*1000:.2f} ms")
print(f"平均 FPS: {1/avg_time:.1f}")
EOF
```

### 3. RKNN 模拟器验证

```bash
# 用 PC 模拟器测试 RKNN 模型
python3 scripts/run_rknn_sim.py \
    --model artifacts/models/person_yolo11s_416.rknn \
    --source assets/test.jpg
```

### 4. 精度对比

```bash
# ONNX vs RKNN 精度对比
python3 scripts/compare_onnx_rknn.py \
    --onnx runs/detect/person_yolo11s_416/weights/best.onnx \
    --rknn artifacts/models/person_yolo11s_416.rknn
```

---

## 🔄 更新项目文件

训练完成后，需要更新以下文件：

### 1. 更新模型链接

```bash
# 将新模型设为默认模型
cd artifacts/models
ln -sf ../../runs/detect/person_yolo11s_416/weights/best.pt best.pt
ln -sf ../../runs/detect/person_yolo11s_416/weights/best.onnx best.onnx
ln -sf person_yolo11s_416.rknn best.rknn
```

### 2. 更新配置文件

编辑 `config/detection/detect.yaml`：
```yaml
model:
  type: yolo11s  # 更新模型类型
  weights: artifacts/models/best.pt
  num_classes: 1  # 只有一个类别: person
  class_names: ['person']
```

### 3. 更新论文数据

编辑论文文件，更新：
- **mAP@0.5**: [新的验证结果]
- **mAP@0.5:0.95**: [新的验证结果]
- **模型**: YOLO11s (9.4M 参数)
- **检测类别**: 1 (专注行人检测)

---

## 📈 预期改进

| 指标 | 原模型 (yolo11n, 80类) | 新模型 (yolo11s, 1类) | 改进 |
|------|----------------------|---------------------|------|
| **mAP@0.5** | 86.14% | **预期 92-95%** | +6-9% |
| **mAP@0.5:0.95** | 61.28% | **预期 68-72%** | +7-11% |
| **参数量** | 2.6M | 9.4M | +3.6× |
| **推理延迟 (PC ONNX)** | 41.6ms | **预期 50-70ms** | +20-70% |
| **推理延迟 (RK3588 NPU)** | 未测 | **预期 30-50ms** | N/A |

---

## ⚠️ 注意事项

### 1. 磁盘空间

- COCO person 子集: ~20GB (图像)
- 训练输出: ~500MB (模型 + 日志)
- 确保至少有 25GB 可用空间

### 2. 训练时间

- GPU 训练: 2-6 小时
- CPU 训练: 不推荐（需要 20-40 小时）

### 3. 内存需求

- 训练: 至少 8GB RAM
- 推荐: 16GB RAM + 6GB VRAM (GPU)

### 4. 模型大小

- yolo11s.onnx: ~18MB
- yolo11s.rknn (INT8): **预期 9-10MB** (超过 5MB 要求)

**如果 rknn 模型超过 5MB：**
- 选项 1: 继续用 yolo11n (4.7MB, 但 mAP 只有 86%)
- 选项 2: 和老师说明，行人检测需要更大模型以达到 90% mAP
- 选项 3: 尝试模型剪枝（需要额外工作）

---

## 🚀 快速开始

```bash
# 一键训练 yolo11s 行人检测模型
bash scripts/train_person_detector.sh yolo11s

# 等待 2-4 小时训练完成
# 查看结果
tail runs/detect/person_yolo11s_416/results.csv

# 如果 mAP@0.5 < 90%，再试 yolo11m
bash scripts/train_person_detector.sh yolo11m
```

---

## 📝 故障排查

### 问题 1: CUDA out of memory
**解决**: 减小 batch size
```bash
# 编辑 scripts/train_person_detector.sh
BATCH=8  # 从 16 改为 8
```

### 问题 2: 数据集准备失败
**解决**: 检查 COCO 数据集路径
```bash
ls datasets/coco/train2017/ | head
ls datasets/coco/annotations/
```

### 问题 3: 训练太慢
**解决**:
- 确保使用 GPU: `nvidia-smi`
- 减少 epochs: `EPOCHS=50`
- 用更小的模型: `yolo11n`

---

## ✅ 成功标准

训练成功的标志：
- ✅ mAP@0.5 ≥ 90%
- ✅ PC ONNX 推理 < 100ms
- ✅ ONNX vs RKNN 精度差异 < 2%
- ✅ 模型可以成功部署到 RK3588

---

**准备好了吗？运行：**
```bash
bash scripts/train_person_detector.sh yolo11s
```

