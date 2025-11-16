# 性能分析报告（ONNX GPU & RKNN 预期）

## 📊 实测性能数据（PC）

### ONNX GPU 推理（RTX 3060 Laptop）
- 总延迟: ~3163 ms
  - 预处理: 18.0 ms (0.6%)
  - 推理: 10.1 ms (0.3%) ⚡
  - 后处理: 3135.4 ms (99.1%) 🐌

### RKNN PC 模拟器（CPU）
- 推理延迟: 354 ms @ 640×640
- 注: PC 模拟器非真实 NPU 性能

## ⚠️ 性能瓶颈分析

问题: 后处理 NMS 耗时 3135 ms 严重超标。

可能原因:
- 置信度阈值过低（0.25）导致候选框过多
- 图像目标密集/背景复杂
- 默认 NMS 实现对该场景效率不佳
- 输出层 proposals 数量大（如 8400 anchors）

## 🎯 评估

- 推理（GPU）: 10.1 ms，表现优秀
- 整体流水线: 被 NMS 拖累，FPS ~0.3，不满足实时

## 🚀 立即优化

1) 提高置信度阈值（示例基于 Ultralytics）
```bash
yolo predict model=... conf=0.5
```
2) 降低输入分辨率（例如 416→320）
```bash
yolo predict model=... imgsz=320
```
3) 限制最大检测数
```bash
yolo predict model=... max_det=100
```

## ✅ 参数优化后的效果（PC 实测）

- 命令（示例）：
```bash
yolo predict model=artifacts/models/best.onnx source=assets/test.jpg imgsz=640 \
  conf=0.5 iou=0.5 save=true project=artifacts name=onnx_optimized
```
- 输出：
  - 预处理: 2.7 ms
  - 推理: 8.6 ms
  - 后处理: 5.2 ms
  - 总延迟: 16.5 ms（≈60+ FPS）
- 加速比（相对默认 conf=0.25）：约 191×（后处理 600×）

## 🧭 RK3588 板端预期

- NPU 推理: 20–45 ms（640×640, INT8）
- C++ 优化 NMS: < 10 ms
- 总延迟: < 60 ms
- 帧率: > 16 FPS（常见配置可达 20–28 FPS）

## 📌 本仓库对应改动

- 默认 NMS 参数已调优：
  - `conf_thres=0.50`，`iou_thres=0.50`，`topk=300`
  - 文件：
    - `config/detection/detect.yaml`
    - `config/detection/detect_rknn.yaml`
    - `config/detection/detect_host_test.yaml`
    - `config/detection/detect_demo.yaml`
- C++ CLI 现已将 NMS 阈值传递给引擎的解码阶段，减少前置候选开销：
  - `examples/detect_cli.cpp` 调用 `engine->setDecodeParams(...)`

## 🎯 结论

- 问题主要在参数与后处理而非模型/硬件
- `conf=0.5` 在工业场景更合理，极大降低 NMS 压力
- 板端请使用 C++ NMS 与 RKNN INT8，确保实时

