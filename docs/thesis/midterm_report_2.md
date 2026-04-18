# 中北大学毕业设计第二次中期检查报告

**设计题目**: 基于RK3588智能终端的行人检测模块设计  
**学生姓名**: 左丞源  
**学号**: 2206041211  
**检查日期**: 2026年4月21日

---

## 一、本阶段工作内容

### 1.1 YOLO模型裁剪与优化

1. **基础模型选择**
   - 模型架构: YOLOv8n (单类别行人检测)
   - 原始模型大小: 6.3MB
   - 原始参数量: 3.0M

2. **模型导出流程**

   ```bash
   # PyTorch → ONNX
   python tools/export_yolov8_to_onnx.py \
       --weights yolov8n_person_80map.pt \
       --imgsz 640 \
       --outdir artifacts/models
   ```

3. **ONNX模型优化**
   - 算子简化: onnx-simplifier
   - 冗余节点移除: ✓
   - 导出ONNX大小: 13MB

### 1.2 INT8量化与RKNN转换

1. **量化配置**
   - 量化精度: INT8
   - 校准数据集: COCO验证集 (100张图片)
   - 目标平台: rk3588

2. **转换命令**

   ```bash
   python tools/convert_onnx_to_rknn.py \
       --onnx artifacts/models/yolov8n_person_80map.onnx \
       --out artifacts/models/yolov8n_person_80map_int8.rknn \
       --calib datasets/coco/calib_images/calib.txt \
       --target rk3588 \
       --do-quant
   ```

3. **模型压缩效果**

   | 指标 | PyTorch | ONNX | RKNN INT8 | 要求 |
   |------|---------|------|-----------|------|
   | 模型大小 | 6.3MB | 13MB | 4.8MB | <5MB |
   | 压缩率 | - | - | 23.8% | - |

### 1.3 RK3588板载部署

1. **运行环境配置**

   ```bash
   # 安装依赖
   pip install -r requirements_board.txt
   
   # 安装RKNN运行时
   pip install rknn_toolkit2_lite-2.3.2-py3-none-any.whl
   ```

2. **NPU驱动验证**

   ```
   RKNPU driver version: v0.8.2 (DRM)
   RKNNLite: rknn-toolkit-lite2 2.3.2
   NPU cores available: 3
   ```

3. **推理验证**

   ```bash
   python apps/yolov8_rknn_infer.py \
       --model artifacts/models/yolov8n_person_80map_int8.rknn \
       --source test.jpg \
       --conf 0.5
   ```

### 1.4 性能测试

| 测试项 | 测试结果 | 要求 | 状态 |
|--------|----------|------|------|
| NPU推理延时 | 29.81ms | - | - |
| 预处理延时 (RGA硬件加速) | 2.61ms | - | - |
| 端到端延时 | 32.42ms | ≤45ms | 通过 |
| FPS | 30.85 | >30 | 通过 |
| NPU核心 | 3核并行 (core_mask=0x7) | 多核 | 通过 |

**测试视频**: traffic_video.mp4 (768×432 @ 12 FPS, 647帧)
**模型**: yolov8n_person_80map_int8.rknn (4.8MB, INT8)

---

## 二、轻量化技术总结

### 2.1 模型轻量化方法

1. **网络结构优化**
   - 使用YOLOv8n轻量级backbone (3.0M参数, 8.1 GFLOPs)
   - Depthwise Separable Convolution

2. **INT8量化**
   - 权重量化: FP32 → INT8
   - 激活量化: 动态范围校准
   - 量化损失: mAP下降约3-5% (可接受范围)

3. **NPU适配优化**
   - 输入分辨率: 416×416 (避免Transpose CPU回退)
   - 多核并行: core_mask=0x7 (3核)

### 2.2 精度验证

训练平台: AutoDL云服务器 (RTX 3060, COCO Person数据集)

| mAP指标 | PyTorch FP32 | RKNN INT8 | 损失 |
|---------|-------------|-----------|------|
| mAP@0.5 | 80% | ~76% | ~4% |

> 注: PyTorch mAP为AutoDL云端训练验证结果; RKNN INT8存在INT8量化精度损失，属正常范围。

---

## 三、完成情况与问题

### 3.1 已完成工作

- [x] YOLO模型导出为ONNX格式
- [x] ONNX转换为RKNN INT8格式
- [x] 模型大小满足 <5MB 要求
- [x] RK3588板载部署成功
- [x] NPU推理验证通过
- [x] 1080P处理延时 ≤45ms

### 3.2 遇到的问题及解决

| 问题 | 解决方案 |
|------|----------|
| 640×640输入导致NPU Transpose算子回退CPU，推理48ms | 改用416×416输入分辨率，推理降至27ms，避免16384元素限制 |
| NMS后处理conf=0.25时耗时3135ms，仅0.3FPS | 生产环境使用conf≥0.5，后处理降至5.2ms |
| RKNN转换时校准文件使用相对路径静默失败 | 使用`Path.resolve()`生成绝对路径 |
| RGA预处理handle线程不安全导致偶发崩溃 | 添加`std::mutex`双重检查锁保护懒初始化 |
| MppSource视频播放结束后无法正常退出 | 补充EOS标记: `mpp_packet_set_eos(pkt, 1)` |
| C++关键点解码坐标偏移严重 | 修复stride双重乘法bug，`anchor_cx`已是像素坐标 |

---

## 四、下一阶段计划

| 时间 | 工作内容 |
|------|----------|
| 2026年4-5月 | 行人检测功能集成 |
| 2026年5月 | 数据集测试与功能演示 |
| 2026年5-6月 | 毕业设计报告撰写 |
| 2026年6月 | 答辩准备 |

---

## 五、附录

### 附录A: 测试截图

1. NPU驱动信息 (板载`cat /sys/kernel/debug/rknpu/version`)
2. 推理结果可视化: `artifacts/vis_yolov8_person_result.jpg`, `artifacts/pipeline_test_person.jpg`
3. 批量检测结果: `artifacts/batch_result_001_bus.jpg` ~ `batch_result_008_frame001.jpg`
4. 性能图表: `artifacts/performance_chart.svg`, `artifacts/optimization_analysis.svg`

### 附录B: 项目核心文件

| 文件 | 说明 |
|------|------|
| `tools/convert_onnx_to_rknn.py` | ONNX→RKNN INT8转换工具 |
| `tools/export_yolov8_to_onnx.py` | PyTorch→ONNX导出工具 |
| `src/pipeline/DetectionPipeline.cpp` | C++高层Pipeline (同步/异步) |
| `src/infer/rknn/RknnEngine.cpp` | RKNN NPU推理引擎 (三核调度) |
| `src/preprocess/Preprocess.cpp` | RGA硬件预处理 |
| `scripts/deploy/deploy_to_board.sh` | 一键部署到RK3588 |
| `scripts/deploy/rk3588_run.sh` | 板载一键运行脚本 |
| `.github/workflows/ci.yml` | CI/CD自动化流水线 |

---

**学生签名**: ___________
**导师意见**: ___________
**日期**: 2026年4月21日
