# 用户指南文档

本目录包含RK3588行人检测系统的用户操作指南和快速开始文档。

## 🚀 快速开始

| 文件 | 说明 | 适用场景 |
|------|------|----------|
| [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) | 快速开始指南 | 首次使用 |
| [QUICK_START_PHASE2.md](QUICK_START_PHASE2.md) | 快速开始 Phase 2 | 进阶使用 |

## 🔧 硬件集成

| 文件 | 说明 | 内容 |
|------|------|------|
| [HARDWARE_INTEGRATION_MANUAL.md](HARDWARE_INTEGRATION_MANUAL.md) | 硬件集成手册 | RK3588硬件配置、接线、部署 |

## 🎓 训练指南

| 文件 | 说明 | 数据集 |
|------|------|--------|
| [PERSON_TRAINING_GUIDE.md](PERSON_TRAINING_GUIDE.md) | 行人检测训练指南 | 通用行人数据集 |
| [CITYPERSONS_QUICKSTART.md](CITYPERSONS_QUICKSTART.md) | CityPersons快速开始 | CityPersons专用 |

## 📖 使用说明

### 第一次使用

1. **环境准备**: 参考 [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)
   ```bash
   # 安装基础运行时依赖（轻量）
   pip install -r requirements.txt

   # 训练/导出/ONNX 相关能力（按需）
   pip install -r requirements_train.txt

   # 验证 ONNXRuntime（训练/导出环境）
   python -c "import onnxruntime; print(onnxruntime.get_device())"
   ```

2. **模型转换**:
   ```bash
   # PyTorch → ONNX
   python tools/export_yolov8_to_onnx.py --weights yolo11n.pt

   # ONNX → RKNN
   python tools/convert_onnx_to_rknn.py --onnx artifacts/models/yolo11n.onnx
   ```

3. **PC端测试**:
   ```bash
   # ONNX推理
   yolo predict model=artifacts/models/yolo11n.onnx source=test.jpg
   ```

### 硬件部署

参考 [HARDWARE_INTEGRATION_MANUAL.md](HARDWARE_INTEGRATION_MANUAL.md)：

```bash
# 一键部署脚本
./scripts/deploy/rk3588_run.sh --model artifacts/models/yolo11n_int8.rknn
```

### 训练自定义模型

#### 方案1: 行人检测（通用）

参考 [PERSON_TRAINING_GUIDE.md](PERSON_TRAINING_GUIDE.md)

```bash
# 使用COCO person子集
bash scripts/datasets/prepare_coco_person.sh

# 开始训练
bash scripts/train/train_pedestrian.sh
```

#### 方案2: CityPersons（高精度）

参考 [CITYPERSONS_QUICKSTART.md](CITYPERSONS_QUICKSTART.md)

```bash
# 准备CityPersons数据集
bash scripts/datasets/prepare_citypersons.py

# 开始训练（目标mAP≥90%）
bash scripts/train/train_citypersons.sh
```

## 🎯 常见任务速查

| 任务 | 命令 | 指南 |
|------|------|------|
| PC推理测试 | `yolo predict model=*.onnx` | QUICK_START_GUIDE |
| 板端部署 | `scripts/deploy/rk3588_run.sh` | HARDWARE_INTEGRATION_MANUAL |
| 模型训练 | `bash scripts/train/*.sh` | PERSON_TRAINING_GUIDE |
| 性能测试 | `bash scripts/run_bench.sh` | QUICK_START_PHASE2 |
| 网络配置 | `scripts/network/*.sh` | HARDWARE_INTEGRATION_MANUAL |

## 📊 性能基准参考

- **PC ONNX GPU**: 8.6ms @ 416×416 (RTX 3060)
- **RK3588 NPU预期**: 20-30ms @ 416×416 (INT8量化)
- **网络吞吐量**: ≥900Mbps (双千兆网口)
- **目标FPS**: >30 FPS

## 🔗 相关文档

- **技术深度文档**: [../driver/](../driver/) (设备树配置、驱动适配)
- **毕业论文**: [../thesis/](../thesis/) (开题报告、论文章节)
- **项目报告**: [../reports/](../reports/) (状态报告、评审文档)
- **主文档**: [../../README.md](../../README.md)
- **AI助手指南**: [../../CLAUDE.md](../../CLAUDE.md)

## ❓ 常见问题

### Q: 如何验证环境配置正确？
A: 运行 `python scripts/check_paths.py`

### Q: 模型转换失败怎么办？
A: 检查 ONNX opset版本，参考 QUICK_START_GUIDE.md 的故障排除章节

### Q: 板端部署提示库文件缺失？
A: 参考 HARDWARE_INTEGRATION_MANUAL.md 的依赖安装部分

---

**最后更新**: 2025-11-19
**维护**: RK3588项目团队
