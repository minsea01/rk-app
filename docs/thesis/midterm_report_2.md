# 中北大学毕业设计第二次中期检查报告

**设计题目**: 基于RK3588智能终端的行人检测模块设计  
**学生姓名**: 左丞源  
**学号**: 2206041211  
**检查日期**: 2026年4月 ___日

---

## 一、本阶段工作内容

### 1.1 YOLO模型裁剪与优化

1. **基础模型选择**
   - 模型架构: YOLO11n / YOLOv8n
   - 原始模型大小: ___MB
   - 原始参数量: ___M

2. **模型导出流程**

   ```bash
   # PyTorch → ONNX
   python tools/export_yolov8_to_onnx.py \
       --weights yolo11n.pt \
       --imgsz 640 \
       --outdir artifacts/models
   ```

3. **ONNX模型优化**
   - 算子简化: onnx-simplifier
   - 冗余节点移除: ✓
   - 导出ONNX大小: ___MB

### 1.2 INT8量化与RKNN转换

1. **量化配置**
   - 量化精度: INT8
   - 校准数据集: COCO验证集 (___张图片)
   - 目标平台: rk3588

2. **转换命令**

   ```bash
   python tools/convert_onnx_to_rknn.py \
       --onnx artifacts/models/yolo11n.onnx \
       --out artifacts/models/yolo11n.rknn \
       --calib datasets/coco/calib_images/calib.txt \
       --target rk3588 \
       --do-quant
   ```

3. **模型压缩效果**

   | 指标 | PyTorch | ONNX | RKNN INT8 | 要求 |
   |------|---------|------|-----------|------|
   | 模型大小 | ___MB | ___MB | ___MB | <5MB |
   | 压缩率 | - | - | ___% | - |

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
   RKNPU driver version: ___
   NPU cores available: 3
   ```

3. **推理验证**

   ```bash
   python apps/yolov8_rknn_infer.py \
       --model artifacts/models/yolo11n.rknn \
       --source test.jpg \
       --conf 0.5
   ```

### 1.4 性能测试

| 测试项 | 测试结果 | 要求 | 状态 |
|--------|----------|------|------|
| 单帧推理延时 | ___ms | - | - |
| 预处理延时 | ___ms | - | - |
| 后处理延时 | ___ms | - | - |
| 端到端延时 (1080P) | ___ms | ≤45ms | [通过/未通过] |
| FPS | ___ | >30 | [通过/未通过] |
| NPU利用率 | ___% | - | - |

**测试图片分辨率**: 1920×1080 (1080P)

---

## 二、轻量化技术总结

### 2.1 模型轻量化方法

1. **网络结构优化**
   - 使用YOLO11n轻量级backbone
   - Depthwise Separable Convolution

2. **INT8量化**
   - 权重量化: FP32 → INT8
   - 激活量化: 动态范围校准
   - 量化损失: mAP下降 ___% (可接受范围)

3. **NPU适配优化**
   - 输入分辨率: 416×416 (避免Transpose CPU回退)
   - 多核并行: core_mask=0x7 (3核)

### 2.2 精度验证

| mAP指标 | ONNX FP32 | RKNN INT8 | 损失 |
|---------|-----------|-----------|------|
| mAP@0.5 | ___% | ___% | ___% |
| mAP@0.5:0.95 | ___% | ___% | ___% |

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
| [填写问题1] | [填写解决方案] |
| [填写问题2] | [填写解决方案] |

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

[在此插入截图]

1. NPU驱动信息
2. 推理结果可视化
3. 性能测试数据

### 附录B: 项目核心文件

| 文件 | 说明 |
|------|------|
| `tools/convert_onnx_to_rknn.py` | RKNN转换工具 |
| `apps/yolov8_rknn_infer.py` | 板载推理脚本 |
| `scripts/deploy/rk3588_run.sh` | 一键运行脚本 |

---

**学生签名**: ___________  
**导师意见**: ___________  
**日期**: 2026年4月___日
