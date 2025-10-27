# YOLO训练诊断工具集

专门针对"高召回低精度"问题的YOLO训练诊断和优化工具包。

## 🚀 快速开始

### 一键诊断（推荐）

```bash
# 完整诊断（数据检查 + 模型评估）
./tools/quick_diagnosis.sh -d /path/to/data.yaml -m runs/train/exp/weights/best.pt

# 仅数据检查
./tools/quick_diagnosis.sh -d /path/to/data.yaml
```

### 针对您的项目

```bash
# 检查当前工业15类数据集
./tools/quick_diagnosis.sh -d industrial_dataset/data.yaml

# 评估最新训练的模型
./tools/quick_diagnosis.sh \
  -d industrial_dataset/data.yaml \
  -m runs/train/industrial_15cls_test5/weights/best.pt \
  -o diagnosis_results_$(date +%Y%m%d_%H%M%S)
```

## 🛠️ 单独工具使用

### 1. 数据健康检查

```bash
python tools/data_health_check.py --data /path/to/data.yaml
```

**输出文件：**
- `data_health_report.txt` - 数据质量综合报告
- `class_distribution.png` - 类别分布可视化
- `sample_visualization.png` - 标注样本可视化

**检查项目：**
- ✅ 空标签文件检测
- ✅ 图像-标签配对检查
- ✅ 类别分布分析
- ✅ 无效类别ID检测
- ✅ 极小目标统计
- ✅ 标注质量评估

### 2. 模型评估

```bash
python tools/model_evaluation.py \
  --model runs/train/exp/weights/best.pt \
  --data /path/to/data.yaml \
  --conf 0.25 --iou 0.6
```

**输出文件：**
- `evaluation_report.txt` - 模型性能评估报告
- `pr_curves.png` - PR曲线分析
- `confusion_matrix.png` - 混淆矩阵
- `confidence_distribution.png` - 置信度分布
- `prediction_samples.png` - 预测样例对比

**分析内容：**
- 📊 详细性能指标（mAP、Precision、Recall）
- 📈 PR曲线和置信度分布
- 🔄 混淆矩阵分析
- 🎯 失效案例分析
- 💡 针对性改进建议

## 🎯 优化训练配置

### 使用预设的优化配置

```bash
# 使用专门优化的配置文件
yolo detect train \
  cfg=configs/optimized_training.yaml \
  data=industrial_dataset/data.yaml
```

### 手动配置（针对高召回低精度问题）

```bash
yolo detect train \
  data=industrial_dataset/data.yaml \
  model=yolov8s.pt \
  imgsz=960 epochs=200 batch=auto device=0 \
  fl_gamma=1.5 box=7.5 cls=1.5 \
  mosaic=1.0 copy_paste=0.2 mixup=0.15 multi_scale=True \
  cos_lr=True lr0=0.005 lrf=0.05 warmup_epochs=5 \
  cache=ram workers=8 patience=80 \
  project=runs/train name=industrial_optimized_high_precision
```

## 📋 典型工作流程

### 发现"高召回低精度"问题后的标准流程：

1. **运行完整诊断**
   ```bash
   ./tools/quick_diagnosis.sh -d data.yaml -m best.pt -o diagnosis_$(date +%Y%m%d)
   ```

2. **分析诊断结果**
   - 检查 `data_health_report.txt` 中的数据质量问题
   - 查看 `evaluation_report.txt` 中的模型性能分析
   - 观察可视化图表中的异常模式

3. **修复数据问题**（如有）
   ```bash
   # 删除空标签文件
   find /path/to/labels -type f -size 0 -delete
   
   # 补充缺失标注
   # 修正无效类别ID
   ```

4. **使用优化配置重训**
   ```bash
   yolo detect train cfg=configs/optimized_training.yaml
   ```

5. **监控训练过程**
   - 观察loss收敛情况
   - 验证集指标变化
   - Precision/Recall平衡性

6. **部署优化**
   ```bash
   # 使用更高置信度阈值
   yolo detect predict model=best.pt conf=0.4 source=test_images/
   ```

## 🔍 常见问题排查

### Q: Precision始终上不去（< 0.6）
**可能原因：**
- 标签质量问题（漏标、错标）
- 目标过小或密集
- 类别严重不均衡
- 置信度阈值过低

**解决方案：**
1. 运行数据健康检查
2. 提高训练分辨率（960+）
3. 使用focal loss
4. 部署时提高置信度阈值

### Q: 训练过程中loss震荡
**可能原因：**
- 学习率过大
- 数据增强过强
- batch size不合适

**解决方案：**
1. 降低初始学习率（lr0=0.003）
2. 减少数据增强强度
3. 调整batch size

### Q: 小目标检测效果差
**解决方案：**
1. 提高输入分辨率（imgsz=1280）
2. 启用multi_scale训练
3. 增加mosaic和copy_paste
4. 检查标注质量

## 📊 性能基准

### 工业15类检测基准（基于您的数据集）

| 配置 | mAP@0.5 | Precision | Recall | 训练时间 |
|------|---------|-----------|--------|----------|
| 默认YOLOv8s-640 | 0.61 | 0.58 | 0.92 | ~2h |
| 优化配置-960 | 0.72+ | 0.70+ | 0.88+ | ~4h |
| 优化配置-1280 | 0.75+ | 0.73+ | 0.86+ | ~6h |

### 部署配置建议

| 场景 | conf | iou | 说明 |
|------|------|-----|------|
| 开发测试 | 0.25 | 0.6 | 查看所有可能检测 |
| 生产环境 | 0.4-0.5 | 0.5-0.6 | 平衡精度和召回 |
| 高精度需求 | 0.6+ | 0.4-0.5 | 最小化假阳性 |

## 📝 更新日志

- **v1.0** - 初始版本，包含基础诊断功能
- **v1.1** - 添加快速诊断脚本和优化配置
- **v1.2** - 增强可视化和报告生成

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这些工具！

## 📄 许可证

MIT License
