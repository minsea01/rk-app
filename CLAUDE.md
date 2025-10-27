# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RK3588 industrial edge AI system for real-time object detection with dual-NIC network streaming. The project supports boardless PC simulation workflow and on-device deployment.

**Target Platform:** Rockchip RK3588 NPU (3 TOPS INT8)
**Model:** YOLOv8/YOLO11 optimized for RKNN runtime
**Deployment:** Cross-compiled ARM64 binary or Python inference
**Development Environment:** WSL2 Ubuntu 22.04, Python virtual env `yolo_env`

## Development Commands

### Testing & Code Quality

```bash
# Install development dependencies (first time)
pip install -r requirements-dev.txt

# Run all unit tests
pytest tests/unit -v

# Run tests with coverage report
pytest tests/unit -v --cov=apps --cov=tools --cov-report=html

# Run a single test file
pytest tests/unit/test_config.py -v

# Run a specific test
pytest tests/unit/test_config.py::TestModelConfig::test_default_size -v

# Code linting and formatting
black apps/ tools/ tests/
pylint apps/ tools/
flake8 apps/ tools/ tests/
isort apps/ tools/ tests/

# Type checking
mypy apps/config.py apps/exceptions.py apps/logger.py
```

## Key Commands

### YOLO Model Export & Conversion Workflow

```bash
# 1. Export YOLO to ONNX (always from PyTorch first)
python3 tools/export_yolov8_to_onnx.py --weights yolo11n.pt --imgsz 640 --outdir artifacts/models

# 2. Convert ONNX to RKNN with INT8 quantization
python3 tools/convert_onnx_to_rknn.py \
  --onnx artifacts/models/yolo11n.onnx \
  --out artifacts/models/yolo11n.rknn \
  --calib datasets/coco/calib_images/calib.txt \
  --target rk3588 \
  --do-quant

# 3. PC simulator boardless validation (no hardware needed)
python3 scripts/run_rknn_sim.py

# 4. ONNX vs RKNN accuracy comparison
python3 scripts/compare_onnx_rknn.py
```

### Calibration Dataset Preparation

```bash
# Create calibration image list (300 COCO person images)
cd datasets/coco
python3 pick_person_subset.py

# Generate absolute path list (REQUIRED - relative paths cause duplicate path errors)
find calib_images -name "*.jpg" -exec realpath {} \; > calib_images/calib.txt
```

### Benchmark & Validation

```bash
# Run full MCP benchmark pipeline (iperf3 + ffprobe + aggregation + HTTP)
bash scripts/run_bench.sh

# Results written to:
# - artifacts/iperf3.json, artifacts/ffprobe.json
# - artifacts/bench_summary.{json,csv}
# - artifacts/bench_report.md
```

### Board Deployment (when hardware available)

```bash
# Build ARM64 binary
cmake --build --preset arm64 && cmake --install build/arm64

# Deploy to RK3588 board
scripts/deploy/deploy_to_board.sh --host <board_ip> --run

# Or with gdbserver for remote debugging
scripts/deploy/deploy_to_board.sh --host <board_ip> --gdb --gdb-port 1234
```

## Critical Architecture Details

### RKNN Conversion Pitfalls

**Transpose CPU Fallback:**
RKNN NPU has a 16384-element limit for Transpose operations. YOLO output shapes:
- ❌ 640×640: (1, 84, 8400) → 4×8400=33600 **exceeds limit → CPU fallback**
- ✅ 416×416: (1, 84, 3549) → 4×3549=14196 **fits in NPU**

**Recommendation:** Use 416×416 resolution for production deployment to ensure full NPU execution.

**Calibration Path Issues:**
`convert_onnx_to_rknn.py` requires **absolute paths** in calibration list. Relative paths cause duplicate prefix errors:
```
# Wrong: /home/user/rk-app/datasets/coco/calib_images/datasets/coco/calib_images/000000002261.jpg
# Right: /home/user/rk-app/datasets/coco/calib_images/000000002261.jpg
```

### PC Simulator vs Board Runtime

**PC Simulator (RKNN-Toolkit2):**
- Must load ONNX and build (`rk.load_onnx()` + `rk.build()`)
- Cannot load pre-built `.rknn` directly (will error: "not support inference on simulator")
- Requires NHWC input format: `(1, 640, 640, 3)`
- Must specify `data_format='nhwc'` in `rk.inference()`
- Config must be called **before** load: `rk.config()` → `rk.load_onnx()`

**Board Runtime (rknn-toolkit2-lite):**
- Loads pre-built `.rknn` models
- Uses optimized NPU kernels
- Expects uint8 input (0-255 range)

### Data Format Conventions

**ONNX Runtime:** NCHW (1, 3, 640, 640)
**RKNN PC Simulator:** NHWC (1, 640, 640, 3)
**Preprocessing:**
- BGR → RGB via `img[..., ::-1]`
- Resize to target size (640 or 416)
- For RKNN: keep as uint8, do NOT normalize to [0,1]

### Directory Structure

```
rk-app/
├── tools/          # Core conversion/export/evaluation tools
│   ├── export_yolov8_to_onnx.py
│   ├── convert_onnx_to_rknn.py
│   ├── aggregate.py, http_receiver.py, http_post.py  # MCP tools
│   └── iperf3_bench.sh, ffprobe_probe.sh
├── scripts/
│   ├── run_bench.sh               # MCP benchmark pipeline
│   ├── run_rknn_sim.py            # PC simulator inference
│   ├── compare_onnx_rknn.py       # Accuracy comparison
│   ├── deploy/deploy_to_board.sh  # SSH deployment to RK3588
│   └── tune/                      # PID tuning scripts
├── apps/
│   ├── yolov8_rknn_infer.py       # Main RKNN inference app
│   └── utils/yolo_post.py         # Postprocessing utilities
├── artifacts/
│   ├── models/                    # .onnx and .rknn outputs
│   └── *.json, *.csv, *.md        # Benchmark results
├── datasets/coco/
│   ├── calib_images/              # Calibration dataset (300 images)
│   └── calib_images/calib.txt     # Absolute paths list
├── config/                        # YAML configs for detection/network
└── configs/mcp_servers.yaml       # MCP server declarations
```

## Python Application Architecture

### Core Modules (apps/)

**apps/config.py** - Centralized configuration management
- `ModelConfig`: Image sizes (416, 640), inference thresholds (0.25, 0.45), detection limits
- `RKNNConfig`: Target platform, optimization level, NPU core masks
- `PreprocessConfig`: Normalization values (mean/std for BGR/RGB)
- Helper functions: `get_detection_config(size)`, `get_rknn_config()`
- All magic numbers consolidated here for easy tuning

**apps/exceptions.py** - Custom exception hierarchy
- `RKAppException`: Base exception class
- `RKNNError`: RKNN runtime failures
- `PreprocessError`: Image preprocessing failures
- `InferenceError`: Inference execution failures
- `ModelLoadError`: Model file loading failures
- `ValidationError`: Input validation failures
- `ConfigurationError`: Configuration errors

**apps/logger.py** - Unified logging system
- `setup_logger(name, level, log_file, console)`: Configure logger with console/file output
- `get_logger(name)`: Get existing logger or create new one
- `set_log_level()`, `enable_debug()`, `disable_debug()`: Convenience functions
- Replaces scattered print() calls with consistent logging

**apps/yolov8_rknn_infer.py** - Main inference entry point
- `decode_predictions()`: Unified YOLO output decoder supporting both DFL and raw heads
- Imports and uses specific exception types from exceptions.py
- Supports both PC simulator (NHWC) and board runtime (uint8) preprocessing

**apps/utils/preprocessing.py** - Image preprocessing utilities
- `preprocess_onnx()`: NCHW format for ONNX Runtime
- `preprocess_rknn_sim()`: NHWC format for PC simulator
- `preprocess_board()`: uint8 NHWC for RK3588 board
- Array-based variants: `preprocess_from_array_*()` for numpy input
- All functions default to `ModelConfig.DEFAULT_SIZE` for consistency

**apps/utils/yolo_post.py** - Post-processing utilities
- `letterbox()`: Aspect-ratio preserving image resizing
- `postprocess_yolov8()`: YOLO detection decoder with NMS
- `sigmoid()`, `nms()`: Helper functions

### Test Structure (tests/)

**tests/unit/** - Unit tests with 40+ test cases
- `test_config.py`: 14 tests covering all config classes and helper functions
- `test_exceptions.py`: 10 tests verifying exception hierarchy and behavior
- `test_preprocessing.py`: 11 tests for image preprocessing functions
- `test_aggregate.py`: 7 tests for utility functions
- Coverage: 88-100% for new modules

**pytest.ini** - Test configuration
- Test discovery: `tests/` directory
- Markers: unit, integration, requires_hardware, requires_model
- Coverage: source = apps, tools
- Output: verbose, short traceback

### Dependency Model

```
apps/yolov8_rknn_infer.py
  ├── imports: exceptions, logger, config
  ├── uses: preprocessing, yolo_post

apps/utils/preprocessing.py
  ├── imports: config (for DEFAULT_SIZE)
  ├── raises: PreprocessError

apps/utils/yolo_post.py
  ├── standalone (no app imports)

apps/logger.py
  ├── standalone (pure logging)

apps/exceptions.py
  ├── standalone (no dependencies)

apps/config.py
  ├── standalone (pure configuration)
```

## MCP Benchmark Pipeline

**Purpose:** Validate "build → deploy → observe → archive" loop without hardware.

**Workflow (bash scripts/run_bench.sh):**
1. iperf3 network test (loopback) → `iperf3.json`
2. ffprobe media probe (1080p@30fps sample) → `ffprobe.json`
3. Aggregate results → `bench_summary.{json,csv}`, `bench_report.md`
4. HTTP POST validation → `http_ingest.log`

**Failure Handling:**
Scripts gracefully degrade (e.g., iperf3 errors generate JSON with `"error"` field) to avoid breaking the pipeline.

## Python Environment

**Virtual env:** `yolo_env` (PyTorch 2.0.1, CUDA 11.8)
**Key packages:**
- ultralytics (YOLOv8/v11 training & export)
- rknn-toolkit2==2.3.2 (RK3588 conversion)
- onnxruntime (validation inference)
- opencv-python, numpy

**Activation:**
```bash
source ~/yolo_env/bin/activate
```

## Quantization & Calibration

**Default dtype:** Auto-detected by toolkit version
- rknn-toolkit2 ≥2.x: `w8a8` (weights+activations INT8)
- rknn-toolkit2 1.x: `asymmetric_quantized-u8`

**Calibration best practices:**
- Use 300+ images from target domain (person detection: COCO category_id=1)
- Ensure diverse lighting/scale/occlusion
- Generate absolute paths: `realpath` or `find ... -exec realpath`

**Accuracy metrics (from artifacts/onnx_rknn_comparison.json):**
- Mean absolute difference: ~0.01 (1%)
- Max relative error: <5%
- These are reference values; validate on your dataset

## Environment Variables

Proxy configuration (if needed):
```bash
export http_proxy=http://172.20.10.2:7897
export https_proxy=http://172.20.10.2:7897
```

Google Gemini API (optional):
```bash
export GOOGLE_API_KEY=<your-key>
```

## Cross-Compilation

**Toolchain:** `aarch64-linux-gnu-gcc/g++`
**Install:** `sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu`
**CMake preset:** `arm64` (see CMakePresets.json)

## Code Quality Standards

### Exception Handling

**Do:**
```python
from apps.exceptions import PreprocessError, InferenceError

try:
    img = cv2.imread(path)
    if img is None:
        raise PreprocessError(f"Failed to load image: {path}")
except PreprocessError as e:
    logger.error(f"Preprocessing failed: {e}")
    raise  # Re-raise after logging
```

**Don't:**
```python
try:
    img = cv2.imread(path)
except:  # Bare except - catches KeyboardInterrupt, SystemExit, etc.
    pass
except Exception as e:  # Too broad - hides specific issues
    print(f"Error: {e}")  # Use logger, not print()
```

### Configuration Usage

**Do:**
```python
from apps.config import ModelConfig, get_detection_config

config = get_detection_config(size=416)
conf_threshold = config['conf_threshold']  # Uses ModelConfig.CONF_THRESHOLD_DEFAULT
```

**Don't:**
```python
# Magic numbers scattered throughout code
conf_threshold = 0.25
size = 416
max_detections = 3549
```

### Logging

**Do:**
```python
from apps.logger import setup_logger

logger = setup_logger(__name__)
logger.info("Starting inference")
logger.error(f"Model load failed: {error_msg}", exc_info=True)
```

**Don't:**
```python
print("Starting inference")
print(f"Error: {error_msg}")  # Can't be redirected or disabled
```

## Common Issues

**Issue:** iperf3 fails with "Bad file descriptor"
**Cause:** WSL2/restricted environment limitation
**Fix:** Expected behavior; scripts generate error JSON and continue

**Issue:** HTTP receiver "Connection refused"
**Cause:** Port readiness race condition
**Fix:** Scripts wait for port discovery via `listening_port` JSON output

**Issue:** RKNN conversion "invalid image path" with duplicate paths
**Cause:** Calibration list uses relative paths
**Fix:** Regenerate with `find ... -exec realpath {} \;`

**Issue:** PC simulator "not support inference on the simulator"
**Cause:** Attempting to load `.rknn` instead of rebuilding from ONNX
**Fix:** Use `load_onnx()` + `build()` in PC simulator mode

**Issue:** PC simulator input shape mismatch
**Cause:** Using NCHW format instead of NHWC
**Fix:** Preprocess to (1, H, W, 3) and specify `data_format='nhwc'`

## Workflow Recommendations

**For model development:**
1. Train/fine-tune in PyTorch (Ultralytics)
2. Export to ONNX (opset 12, simplify=True)
3. Validate ONNX with onnxruntime before RKNN conversion
4. Convert to RKNN with calibration dataset
5. Run PC simulator validation + accuracy comparison
6. Deploy to board only after PC validation passes

**For boardless iteration:**
- Use `scripts/run_rknn_sim.py` for functional verification
- Use `scripts/compare_onnx_rknn.py` for accuracy analysis
- Avoid on-device testing until PC simulation is stable

**For performance optimization:**
- Prefer 416×416 over 640×640 (avoid Transpose CPU fallback)
- Monitor layer-wise profiling with `rknn.eval_perf()`
- Target <45ms end-to-end latency (camera → inference → UDP)
