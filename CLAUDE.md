# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Overview

This is a RK3588-targeted industrial computer vision and embedded control system with cross-compilation support:
- **Object detection pipeline** using YOLO models with ONNX/RKNN inference engines
- **RK3588 NPU acceleration** via RKNN-Toolkit2 with INT8 quantization (24+ FPS @ 640x640)
- **Cross-platform development** (x86 debug + ARM64 release with QEMU testing)
- **PID control systems** with automated tuning and real-time benchmarking
- **Industrial camera integration** (GigE Vision, dual-network architecture)
- **Containerized ML pipeline** for training, export, and quantization

## Key Commands

### C++ Development and Testing
**Primary development workflow**: VS Code with CMake presets

```bash
# x86 development build and run
cmake --preset x86-debug
cmake --build --preset x86-debug
./out/x86/bin/rk_app

# VS Code integration (preferred)
# Ctrl+Shift+B → "x86: run" (builds and executes)
# F5 for debugging with GDB

# ARM64 cross-compilation for RK3588
cmake --preset arm64-release  
cmake --build --preset arm64-release
cmake --install build/arm64

# ARM64 testing with QEMU
# Use VS Code task: Ctrl+Shift+B → "arm64: run (qemu)"

# Unit testing (Google Test framework)
ctest --preset x86-debug
ctest --preset x86-debug --verbose  # Detailed output

# VS Code shortcuts
# Ctrl+Shift+B → "x86: run" (default build task)
# F5 for debugging with GDB
# Ctrl+Shift+P → "Tasks: Run Task" for other build tasks
```

### YOLO Model Pipeline
**Container-based training and conversion workflow**

```bash
# Complete pipeline (uses Docker containers)
make all              # train → export → convert-int8

# Individual steps
make train            # Train YOLOv8 with GPU acceleration
make export           # Export trained model to ONNX
make convert-int8     # Convert ONNX to RKNN with INT8 quantization
make convert-fp16     # Alternative FP16 conversion if INT8 fails

# Custom training parameters (override configs/exp.yaml defaults)
make train IMG_SIZE=640 EPOCHS=100 DEVICE=0 RUN_NAME=my_experiment

# Training with different datasets
DATA_YAML=/path/to/dataset.yaml make train

# Quick testing/prototyping
make train IMG_SIZE=320 EPOCHS=10 BATCH=2  # Fast training for testing

# Available make targets
make calib CALIB_SRC=/path/to/dataset.yaml CALIB_N=300  # Create calibration dataset
make compare COMPARE_IMG=/path/to/test.jpg             # Compare ONNX vs RKNN
make vis IMG=/path/to/image.jpg                        # Visualize inference results
```

### PID Control System
```bash
# Automated PID tuning with visualization
python3 scripts/auto_tune_pid.py --profile lowovershoot --plot-best
python3 scripts/auto_tune_pid.py --profile fast100 --plot-best

# Manual benchmarking
./build/x86-debug/pid_bench --impl=cpp --plant=second --dt=0.001 --steps=5000

# Results output to results/autotune/ with plots
```

## Architecture and Structure

### Computer Vision Pipeline
The system implements a modular, plugin-based object detection pipeline following interface-driven design:

- **Capture Layer** (`include/rkapp/capture/`): Abstracted input sources via `ISource` interface
  - `VideoSource`: Video file processing with OpenCV
  - `GigeSource`: Industrial camera interface (GStreamer + Aravis GigE Vision)
  - `FolderSource`: Batch image processing for datasets
  - Unified interface: `open()`, `read()`, `release()` with metadata access
  
- **Inference Engines** (`include/rkapp/infer/`): Pluggable model backends via `IInferEngine` interface
  - `OnnxEngine`: ONNXRuntime with CPU/GPU execution providers
  - `RknnEngine`: RKNN NPU acceleration with pimpl pattern for build isolation
  - Auto-detection of model metadata (classes, input size, output format)
  - Returns standardized `Detection` objects with bbox, confidence, class info
  
- **Processing Chain** (`include/rkapp/preprocess/`, `include/rkapp/post/`):
  - Letterbox preprocessing with aspect ratio preservation and padding
  - NMS post-processing with configurable confidence/IoU thresholds
  - Coordinate transformation from model space to original image space
  
- **Output Layer** (`include/rkapp/output/`): Configurable result transmission
  - `TcpOutput`: Network transmission with dual-NIC support and interface binding
  - JSON-formatted detection results with timestamps
  - Optional file output for debugging/logging

### Build System Design
**Dual-target CMake system** with extensive preset configuration:

- **CMakePresets.json**: Defines x86-debug and arm64-release configurations
- **Cross-compilation**: Uses `toolchain-aarch64.cmake` for ARM64 targeting
- **Feature toggles**: ENABLE_ONNX, ENABLE_RKNN, ENABLE_RGA options
- **Sanitizers**: ASan/UBSan automatically enabled for x86 debug builds
- **RPATH handling**: Ensures portable deployment with relative library paths

### Container Strategy
**Multi-stage Docker workflow** for reproducible model pipeline:

- **Training container** (`docker/x86-rknn-build.Dockerfile`): Ubuntu 20.04 + Python 3.8 + Ultralytics + RKNN-Toolkit2
- **Runtime container** (`docker/arm64-rknn-run.Dockerfile`): Device deployment with RKNN-Toolkit-Lite2
- **Version pinning**: RKNN toolkit versions (1.7.5) matched to device runtime requirements

### Configuration Management
**Unified YAML configuration** (`configs/exp.yaml`) controls entire pipeline:

```yaml
dataset_yaml: /path/to/data.yaml    # Training dataset
imgsz: 640                          # Model input resolution  
epochs: 100                         # Training duration
target_platform: rk3588            # NPU target
quant_dtype: asymmetric_quantized-u8 # Quantization format
core_mask: 0x7                      # NPU core utilization
```

## Development Workflows

### Model Training and Deployment
1. **Dataset preparation**: Ensure YOLO format dataset with >10 classes for industrial requirements
2. **Training**: Configure `configs/exp.yaml` and run `make train` in Docker
3. **Export**: `make export` creates static ONNX with fixed input shapes
4. **Quantization**: `make convert-int8` generates NPU-optimized RKNN model
5. **Integration**: Deploy RKNN model to `artifacts/models/` for C++ inference

### Cross-Platform Development
1. **Local development**: Use VS Code tasks for x86 builds with full debugging support
2. **ARM64 validation**: QEMU-based testing without hardware dependency  
3. **Device deployment**: `scripts/deploy_to_board.sh` for real hardware testing
4. **Performance profiling**: Built-in timing and NPU core utilization monitoring

### Code Quality Standards
- **Format**: Google C++ style via `.clang-format` (100 character limit)
- **Linting**: Comprehensive clang-tidy checks for bugs, performance, modernization
- **Testing**: Google Test framework with CTest integration
- **Memory safety**: AddressSanitizer enabled for debug builds

## Hardware Integration Notes

### RK3588 NPU Optimization
- **Multi-core utilization**: Configure `core_mask` in exp.yaml (0x7 = all 3 cores)
- **Memory layout**: Models auto-detect NCHW vs NHWC tensor formats  
- **Quantization**: INT8 provides ~4x memory reduction with minimal accuracy loss
- **Performance target**: 24+ FPS for 640x640 input on NPU

### Industrial Camera Integration
- **GigE Vision**: `GigeSource` class handles industrial camera protocols
- **Dual network**: Separate interfaces for capture (cam) and results upload  
- **Real-time constraints**: Frame processing pipeline designed for <42ms latency

## Key Configuration Files
- `configs/exp.yaml`: Unified ML pipeline configuration (training, export, quantization)
- `config/app.yaml`: Runtime application configuration (logging, etc.)
- `config/network/dual_network.yaml`: Dual-NIC network configuration for industrial deployment
- `config/industrial_classes.txt`: 15-class industrial object detection labels
- `CMakePresets.json`: Build configuration presets (x86-debug, arm64-release)
- `toolchain-aarch64.cmake`: Cross-compilation toolchain for RK3588

## Testing and Validation
- **Unit tests**: Google Test framework with CTest integration (`tests/test_*.cpp`)
- **Benchmarking**: PID controller performance testing (`bench/bench_pid.cpp`)
- **QEMU validation**: ARM64 testing without hardware via VS Code tasks
- **Memory safety**: AddressSanitizer enabled for x86 debug builds
- **Code quality**: Clang-format (Google style) and clang-tidy integration

## Deployment Structure
```
out/arm64/                    # Cross-compiled binaries
├── bin/rk_app               # Main application
├── lib/                     # Shared libraries with RPATH
artifacts/models/            # Trained models
├── best.onnx               # Exported ONNX model  
├── best.rknn               # Quantized RKNN model
├── logs/                   # Inference logs and evidence
results/autotune/           # PID tuning outputs with plots
config/                     # Configuration files
├── detection/              # Detection-specific configs
├── network/                # Network configuration
└── deploy/                 # Deployment-specific settings
```

## Model Integration Notes
- **Pre-trained models**: Use `yolov8s.pt` or `yolo11s.pt` as base weights
- **Industrial dataset**: 15 classes (screws, bolts, gears, etc.) available in `/datasets/industrial_15_classes/`
- **Quantization calibration**: Generate calibration dataset with `make calib CALIB_SRC=<dataset.yaml>`
- **Model validation**: Use `make compare` to verify ONNX vs RKNN accuracy
- **Performance monitoring**: Built-in timing measurements and NPU utilization tracking