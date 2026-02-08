# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RK3588 industrial edge AI system for real-time object detection with dual-NIC network streaming. This project is a graduation design for North University of China, focusing on pedestrian detection module design based on RK3588 intelligent terminal.

**Target Platform:** Rockchip RK3588 NPU (6 TOPS, 3×NPU cores, 4×A76+4×A55 CPU, 16GB RAM)
**Model:** YOLOv8/YOLO11 optimized for RKNN runtime with INT8 quantization
**Development:** WSL2 Ubuntu 22.04, Python virtual env `yolo_env`
**Project Quality:** S-Level (95/100)

### Key Requirements

**Technical Specs:**
1. System Migration: Ubuntu 20.04/22.04 on RK3588
2. Dual Gigabit Ethernet: RGMII, ≥900Mbps (Port 1: camera, Port 2: upload)
3. Model Optimization: <5MB, >30 FPS, ≥90% mAP@0.5 on pedestrian detection
4. NPU Deployment: Multi-core parallel processing with RKNN format

**Timeline:** Defense June 2026 (Phase 1 98% complete, Phase 2-4 hardware-dependent)

**Key Metrics:**
- Model size: 4.7MB ✅
- PC performance: 8.6ms @ 416×416 (ONNX GPU)
- mAP baseline: 61.57% (target ≥90% achievable with CityPersons fine-tuning)

## Claude Code Automation

### Slash Commands (`.claude/commands/`)

- **/full-pipeline** - PyTorch → ONNX → RKNN → Validation
- **/thesis-report** - Graduation thesis progress report
- **/performance-test** - ONNX GPU, RKNN sim, MCP benchmarks
- **/board-ready** - RK3588 deployment readiness check
- **/model-validate** - ONNX vs RKNN accuracy comparison

**Output locations:** `artifacts/*_report.md`, `docs/thesis_progress_report_*.md`

See `.claude/commands/README.md` for detailed documentation.

## Key Commands

### C++ 构建与测试

```bash
# 主机调试构建 (ONNX)
cmake --preset x86-debug && cmake --build --preset x86-debug
./build/x86-debug/detect_cli --cfg config/detection/detect.yaml

# 主机 Release 构建（推荐开发用）
cmake --preset native-release && cmake --build --preset native-release

# RK3588 交叉编译 (RKNN)
cmake --preset arm64-release -DENABLE_RKNN=ON && cmake --build --preset arm64
cmake --install build/arm64

# C++ 单元测试
ctest --preset x86-debug
```

**CMake Presets：**
- `x86-debug` / `native-debug` - Debug 构建（自动启用 Sanitizers）
- `native-release` - Release 构建（推荐开发用）
- `arm64-release` - RK3588 交叉编译（需要 sysroot 和 toolchain）
- `x86-debug-nosan` - Debug 构建（禁用 Sanitizers）

**CMake 选项：**
- `-DENABLE_ONNX=ON`（默认）：主机 ONNX 推理，可通过 `ORT_HOME` 指定外部路径
- `-DENABLE_RKNN=ON`：启用 RKNN SDK（默认 `RKNN_HOME=/opt/rknpu2`）
- `-DENABLE_GIGE=ON`：启用 GigE 相机（需要 aravis/gstreamer）
- `-DENABLE_MPP=ON`：启用 MPP 硬件视频解码（RK3588）

### Makefile 快捷命令

```bash
# 完整流水线：训练 → 导出 → 转换
make RUN_NAME=<exp> all MODEL_PREFIX=yolo11n

# 单独目标
make compare COMPARE_IMG=<img>                    # PC vs RKNN 对比
make calib CALIB_SRC=/path/to/data.yaml CALIB_N=300  # 生成校准集
make vis IMG=/path/to/image.jpg                   # 可视化推理
make validate VALIDATE_IMG=assets/test.jpg       # 验证模型
make convert-fp16                                  # 仅 FP16 转换
```

### Model Conversion Workflow

```bash
# 1. Export YOLO to ONNX
python3 tools/export_yolov8_to_onnx.py --weights yolo11n.pt --imgsz 640 --outdir artifacts/models

# 2. Convert to RKNN with INT8 quantization
python3 tools/convert_onnx_to_rknn.py \
  --onnx artifacts/models/yolo11n.onnx \
  --out artifacts/models/yolo11n.rknn \
  --calib datasets/coco/calib_images/calib.txt \
  --target rk3588 \
  --do-quant

# 3. PC simulator validation (boardless)
python3 scripts/run_rknn_sim.py

# 4. Accuracy comparison
python3 scripts/compare_onnx_rknn.py
```

### Python 测试与质量

```bash
# Python 单元测试（跳过需要硬件的测试）
pytest tests/unit -m "not requires_hardware" -v
pytest tests/unit -v --cov=apps --cov=tools --cov-report=html

# 代码质量检查
pre-commit run --all-files  # 或单独运行：
black apps/ tools/ tests/
flake8 apps/ tools/ tests/
```

代码风格、提交规范详见 `AGENTS.md`。

### Calibration Dataset

```bash
# Generate absolute path list (REQUIRED - relative paths cause errors)
cd datasets/coco
find calib_images -name "*.jpg" -exec realpath {} \; > calib_images/calib.txt
```

### mAP Evaluation

```bash
# Evaluate pedestrian mAP (COCO person subset)
python scripts/evaluation/official_yolo_map.py \
  --model artifacts/models/best.pt \
  --annotations datasets/coco/annotations/person_val2017.json \
  --images-dir datasets/coco/val2017 \
  --output artifacts/yolo11n_baseline_map.json

# ONNX vs RKNN comparison
python scripts/evaluation/pedestrian_map_evaluator.py \
  --model-onnx artifacts/models/yolo11n.onnx \
  --model-rknn artifacts/models/yolo11n.rknn \
  --dataset coco_person \
  --output artifacts/map_comparison.json

# Fine-tune on CityPersons (2-4 hours, ≥90% mAP)
bash scripts/train/train_citypersons.sh
```

### Board Deployment

```bash
# Build ARM64 binary
cmake --preset arm64-release -DENABLE_RKNN=ON && cmake --build --preset arm64

# On-device one-click run
scripts/deploy/rk3588_run.sh --model artifacts/models/yolo11n_int8.rknn

# SSH deployment
scripts/deploy/deploy_to_board.sh --host <board_ip> --run
```

### Network Validation

```bash
# RGMII driver configuration (RK3588 board)
sudo bash scripts/network/rgmii_driver_config.sh

# Throughput validation (900Mbps requirement)
bash scripts/network/network_throughput_validator.sh --mode loopback  # PC testing
bash scripts/network/network_throughput_validator.sh --mode simulation  # Theoretical
```

### Performance Benchmarks

```bash
# ONNX GPU inference
source ~/yolo_env/bin/activate
yolo predict model=artifacts/models/best.onnx source=assets/test.jpg imgsz=640 conf=0.5

# Full MCP pipeline
bash scripts/run_bench.sh  # → artifacts/bench_summary.{json,csv}, bench_report.md

# Latency micro-benchmark
python tools/bench_onnx_latency.py --model artifacts/models/best.onnx --runs 50
```

**Performance Findings:**
- ONNX GPU: 8.6ms @ 416×416 (RTX 3060)
- End-to-end optimized: 16.5ms (60+ FPS) with conf=0.5
- ❌ conf=0.25: 3135ms postprocessing → 0.3 FPS (NMS bottleneck)
- ✅ conf=0.5: 5.2ms postprocessing → 60+ FPS (production ready)

## Documentation

### Thesis Documentation (`docs/thesis/`)

**7 complete chapters + opening report (~18,000 words):**
1. Opening Report (开题报告.docx) - background, timeline, technical solution
2. Introduction - research status, contributions, innovation points
3. System Design - hardware/software architecture, module design
4. Model Optimization - INT8 quantization, calibration, conversion toolchain
5. Deployment - Python vs C++, environment setup, one-click scripts
6. Performance Testing - PC benchmarks, RKNN validation, parameter tuning
7. Integration - functional validation, mAP evaluation, compliance analysis
8. Conclusion - achievements, limitations, future work

**Defense Materials:**
- PPT outline (20-25 slides, 12-15 min)
- Speech script (slide-by-slide notes + Q&A guide)

**Workflow Diagrams:** `docs/项目流程框图.md` - 10 Mermaid flowcharts

See `docs/thesis/THESIS_README.md` for complete navigation.

### Technical Guides

- `docs/CONFIG_GUIDE.md` - Configuration priority chain (CLI > ENV > YAML > defaults)
- `docs/docs/RGMII_NETWORK_GUIDE.md` - RGMII driver configuration
- `docs/CITYPERSONS_FINETUNING_GUIDE.md` - Fine-tuning to ≥90% mAP

## Critical Architecture Details

### RKNN Conversion Pitfalls

**Transpose CPU Fallback:**
RKNN NPU has a 16384-element limit for Transpose operations:
- ❌ 640×640: (1, 84, 8400) → 33600 elements **exceeds limit → CPU fallback**
- ✅ 416×416: (1, 84, 3549) → 14196 elements **fits in NPU**

**Recommendation:** Use 416×416 for production to ensure full NPU execution.

**Calibration Path Issues:**
`convert_onnx_to_rknn.py` requires **absolute paths** in calibration list. Relative paths cause duplicate prefix errors.

### PC Simulator vs Board Runtime

**PC Simulator (RKNN-Toolkit2):**
- Must load ONNX + build (`rk.load_onnx()` + `rk.build()`)
- Cannot load pre-built `.rknn` directly
- Requires NHWC input: `(1, 640, 640, 3)`
- Must specify `data_format='nhwc'` in `rk.inference()`
- Config before load: `rk.config()` → `rk.load_onnx()`

**Board Runtime (rknn-toolkit2-lite):**
- Loads pre-built `.rknn` models
- Uses optimized NPU kernels
- Expects uint8 input (0-255 range)

### Data Format Conventions

- **ONNX Runtime:** NCHW (1, 3, 640, 640)
- **RKNN PC Simulator:** NHWC (1, 640, 640, 3)
- **Preprocessing:** BGR → RGB via `img[..., ::-1]`, resize, keep uint8 for RKNN

### Decode Metadata Requirements

- `head=auto` now uses a conservative policy for ambiguous output channels.
- For deterministic decode, provide sidecar metadata:
  - `<model>.json` or `<model>.meta` (preferred)
  - fallback `artifacts/models/decode_meta.json`
- Recommended fields:
  - DFL heads: `head`, `reg_max`, `strides`, `num_classes`
  - RAW heads: `head`, `num_classes`, `has_objectness`

## Project Structure

**C++ 核心流水线：**
- `src/` - 核心实现（capture/preprocess/infer/post/output）
- `include/rkapp/` - 公共头文件
- `examples/` - CLI 示例（`detect_cli.cpp`）

**Python 工具链：**
- `apps/` - 板端 Python Runner（`yolov8_rknn_infer.py` 等）
- `tools/` - 转换/评估工具（`convert_onnx_to_rknn.py` 等）
- `scripts/` - 部署/压测脚本

**其他目录：**
- `.claude/` - 5 个斜杠命令 + 5 个技能
- `artifacts/` - 模型与构建产物
- `config/` - 主配置目录（`configs/` 为符号链接，保持兼容）
- `tests/` - Python + C++ 测试

## C++ Architecture (Critical)

**Interface Hierarchy:**
```
ISource (capture)     → FolderSource, VideoSource, GigeSource, MppSource
IInferEngine (infer)  → OnnxEngine, RknnEngine
IOutput (output)      → TcpOutput, UdpOutput
```

**DetectionPipeline (Pimpl pattern):**
```cpp
// High-level API - src/pipeline/DetectionPipeline.cpp
DetectionPipeline pipeline;
pipeline.init(config);
while (auto result = pipeline.next()) {
    // Process detections
}
```

**Hardware Acceleration Layers:**
- `DmaBuf` - 零拷贝 DMA 缓冲区 (3-5x 内存带宽优化)
- `FramePool` - 预分配帧缓冲池 (避免运行时分配)
- `RGA Preprocess` - 硬件预处理 (0.3ms vs OpenCV 3ms)
- `MPP Decode` - 硬件视频解码 (50% CPU 降低)

**Thread Safety:**
- 所有统计计数器使用 `std::atomic`
- 积压队列使用 `std::mutex` 保护
- `TcpOutput` 提供 `droppedFrames()` / `totalSent()` 监控接口

**Design Patterns:**
- RAII 资源管理 (智能指针 95%+)
- Pimpl 隐藏实现细节
- 工厂函数创建源/引擎 (`createSource()`, `createEngine()`)

**核心模块（Python）：**
- `apps/config.py` - 配置中心（ModelConfig, RKNNConfig）
- `apps/config_loader.py` - 优先级链：CLI > ENV > YAML > defaults
- `apps/exceptions.py` - 自定义异常层次
- `apps/logger.py` - 统一日志接口
- `apps/utils/preprocessing.py` - 图像预处理
- `apps/utils/yolo_post.py` - 后处理（letterbox, NMS）

**关键工具（tools/）：**
- `convert_onnx_to_rknn.py` - ONNX → RKNN 转换（INT8/FP16）
- `export_yolov8_to_onnx.py` - PyTorch → ONNX 导出
- `pc_compare.py` - ONNX vs RKNN 对比验证
- `bench_onnx_latency.py` - ONNX 推理性能基准
- `model_evaluation.py` - 模型 mAP 评估

## Python Environment

**Virtual env:** `yolo_env` (Python 3.10.12, PyTorch 2.0.1+cu117, CUDA 11.7)

```bash
source ~/yolo_env/bin/activate
export PYTHONPATH=$PWD  # Required for apps/ imports
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development only
```

## Training Resources

训练相关文件位于虚拟环境外 (`~/yolo_env/`)，不在项目仓库内。详见 `cloud_training/README.md`。

**Key packages:**
- numpy<2.0 (RKNN toolkit compatibility)
- opencv-python-headless==4.9.0.80
- ultralytics>=8.0.0 (YOLO training & export)
- rknn-toolkit2>=2.3.2 (ONNX→RKNN conversion)
- onnxruntime==1.18.1 (PC validation)
- pytest, black, pylint, flake8, mypy (development)

## Code Quality Standards

**异常处理：** 使用 `apps/exceptions.py` 中的自定义异常（`PreprocessError`, `InferenceError` 等），避免裸 `except` 或过宽的 `except Exception`。

**配置管理：** 使用 `apps/config_loader.py`，优先级链：CLI > ENV > YAML > defaults。避免魔法数字。

**日志：** 使用 `apps/logger.py`，避免 `print()`。

**Git Commit规范：**
- 使用Conventional Commit格式：`feat:`, `fix:`, `docs:`, `refactor:` 等
- ⛔ **禁止使用 `Co-Authored-By` 标签** - 这是个人毕业设计项目
- 例如：`feat: add RKNN INT8 quantization support` ✅

**Pre-commit Hooks (`.pre-commit-config.yaml`)：**
- `black` / `isort` / `flake8` - Python 格式化和检查
- `clang-format` - C++ 格式化 (Google style, 遵循 `.clang-format`)
- `shellcheck` - Shell 脚本检查
- 运行：`pre-commit run --all-files`

## Common Issues

**RKNN conversion "invalid image path"**
→ Calibration list uses relative paths. Regenerate with `find ... -exec realpath {}`

**PC simulator "not support inference"**
→ Loading `.rknn` instead of ONNX. Use `load_onnx()` + `build()` in simulator mode

**PC simulator input shape mismatch**
→ Using NCHW format. Preprocess to (1, H, W, 3) and specify `data_format='nhwc'`

**Configuration conflicts**
→ Use `apps/config_loader.py` with explicit priority: CLI > ENV > YAML > defaults

**Network throughput validation requires hardware**
→ Use loopback mode for toolchain validation or simulation mode for theoretical verification

## Workflow Recommendations

**Model development:**
1. Train/fine-tune in PyTorch (Ultralytics)
2. Export to ONNX (opset 12, simplify=True)
3. Validate ONNX with onnxruntime before RKNN conversion
4. Convert to RKNN with calibration dataset
5. Run PC simulator validation + accuracy comparison
6. Deploy to board only after PC validation passes

**Boardless iteration:**
- Use `scripts/run_rknn_sim.py` for functional verification
- Use `scripts/compare_onnx_rknn.py` for accuracy analysis
- Avoid on-device testing until PC simulation is stable

**Performance optimization:**
- Prefer 416×416 over 640×640 (avoid Transpose CPU fallback)
- Use conf≥0.5 for industrial applications (avoid NMS bottleneck)
- Target <45ms end-to-end latency (camera → inference → UDP)

## Cloud Training (AutoDL 4090)

云端训练脚本位于 `cloud_training/` 目录，用于在 AutoDL 租用 4090 训练 YOLOv8n 行人检测模型。

**训练包文件：**
- `cloud_training.tar.gz` - 打包好的训练脚本
- `cloud_training/setup_autodl.sh` - 环境配置
- `cloud_training/train.sh` - 训练脚本 (100 epochs)
- `cloud_training/export_onnx.sh` - ONNX导出
- `cloud_training/README.md` - 完整使用指南

**AutoDL 配置：**
- GPU: RTX 4090 (24GB)
- 镜像: PyTorch 2.0.0 / Python 3.10 / CUDA 11.8
- 费用: ~¥2.5-3/小时，训练约3小时 ≈ ¥10

**训练流程：**
```bash
# 1. 本地上传到AutoDL
scp -P <端口> cloud_training.tar.gz root@<地址>:~/

# 2. SSH连接后执行
tar -xzf cloud_training.tar.gz && cd cloud_training
bash setup_autodl.sh    # 安装依赖
bash train.sh           # 训练 (2-4小时)
bash export_onnx.sh     # 导出ONNX

# 3. 下载模型回本地
scp -P <端口> root@<地址>:~/pedestrian_training/outputs/yolov8n_pedestrian/weights/best.pt ./artifacts/models/
scp -P <端口> root@<地址>:~/pedestrian_training/outputs/yolov8n_pedestrian/weights/best.onnx ./artifacts/models/

# 4. 本地RKNN转换
python3 tools/convert_onnx_to_rknn.py --onnx artifacts/models/best.onnx --out artifacts/models/yolov8n_pedestrian_int8.rknn --calib datasets/coco/calib_images/calib.txt --target rk3588
```

**预期结果：** mAP ≥90% (CityPersons), 模型 ~4.8MB

## Current Status

**已达成指标：**
| 指标 | 要求 | 实际 | 状态 |
|------|------|------|------|
| 模型大小 | <5MB | 4.3-4.8MB | ✅ |
| 推理速度 | >30 FPS | 40 FPS | ✅ |
| NPU 利用率 | 3核并行 | ✅ | ✅ |
| mAP 精度 | ≥90% | 80% | ⏸️ 待提升 |

**可用模型：** `artifacts/models/`
- `yolo11n_416.rknn` - 4.3MB, 40 FPS
- `yolov8n_person_80map_int8.rknn` - 4.8MB, 80% mAP
- `best.rknn` - 4.7MB, 基线模型

**详细进度：** `artifacts/board_deployment_success_report.md`, `docs/thesis/`
