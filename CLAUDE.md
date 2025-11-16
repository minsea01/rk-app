# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RK3588 industrial edge AI system for real-time object detection with dual-NIC network streaming. This project is a graduation design for North University of China, focusing on pedestrian detection module design based on RK3588 intelligent terminal. The project supports boardless PC simulation workflow and on-device deployment.

**Target Platform:** Rockchip RK3588 NPU (6 TOPS with 3×NPU cores, 4×A76+4×A55 CPU, 16GB RAM, 10W typical power)
**Model:** YOLOv8/YOLO11 optimized for RKNN runtime with INT8 quantization
**Deployment:** Cross-compiled ARM64 binary or Python inference
**Development Environment:** WSL2 Ubuntu 22.04, Python virtual env `yolo_env`

### Graduation Design Requirements

**Project Title:** Pedestrian Detection Module Design Based on RK3588 Intelligent Terminal

**Key Technical Specifications:**
1. **System Migration**: Ubuntu 20.04/22.04 on RK3588 platform
2. **Dual Gigabit Ethernet**: RGMII interface, throughput ≥900Mbps
   - Port 1: Industrial camera connection (1080P real-time capture)
   - Port 2: Detection result upload
3. **Model Optimization**: YOLOv5s/YOLOv8/YOLO11 with INT8 quantization
   - Model size: <5MB
   - FPS: >30
   - Accuracy: mAP@0.5 >90% on pedestrian detection dataset
4. **NPU Deployment**: Multi-core parallel processing with RKNN format

**Timeline Milestones:**
- Phase 1 (Oct-Nov 2025): Literature review + proposal
- Phase 2 (Nov-Dec 2025): System migration + dual-NIC driver development
- Phase 3 (Jan-Apr 2026): Model pruning, optimization, and deployment
- Phase 4 (Apr-Jun 2026): Dataset construction + pedestrian detection implementation
- Defense: June 2026

**Deliverables:**
- Working software package with source code
- Proposal report + 2 progress reports
- Graduation thesis (including English literature translation)
- Live demo system
- Dual-NIC driver implementation
- Pedestrian detection dataset with mAP validation

## Claude Code Automation

**This project includes comprehensive automation via Claude Code slash commands and skills.**

### Slash Commands (in `.claude/commands/`)

Execute complex workflows with simple slash commands:

- **/full-pipeline** - Complete model conversion pipeline (PyTorch → ONNX → RKNN → Validation)
- **/thesis-report** - Generate graduation thesis progress report with compliance analysis
- **/performance-test** - Run comprehensive performance benchmarks (ONNX GPU, RKNN sim, MCP)
- **/board-ready** - Check RK3588 board deployment readiness
- **/model-validate** - Validate model accuracy and compare ONNX vs RKNN

**Usage example:**
```
User: /performance-test
Claude: [Runs ONNX GPU inference, RKNN simulator, MCP benchmarks, generates report]
```

**Output locations:**
- `artifacts/pipeline_report.md` - /full-pipeline output
- `docs/thesis_progress_report_*.md` - /thesis-report output
- `artifacts/performance_report_*.md` - /performance-test output
- `artifacts/board_ready_report.md` - /board-ready output
- `artifacts/validation_report_*.md` - /model-validate output

### Skills (in `.claude/skills/`)

Skills provide detailed workflow definitions that commands execute. Each skill includes:
- Prerequisite checks
- Step-by-step execution plan
- Output validation
- Error handling

See `.claude/commands/README.md` and `.claude/skills/README.md` for detailed documentation.

## Development Commands

### Testing & Code Quality

```bash
# Install development dependencies (first time)
pip install -r requirements-dev.txt

# Run all unit tests (7 test files, 40+ test cases)
pytest tests/unit -v

# Run tests with coverage report (88-100% coverage)
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

### Board Deployment

```bash
# Build ARM64 binary with RKNN support
cmake --preset arm64-release -DENABLE_RKNN=ON && cmake --build --preset arm64 && cmake --install build/arm64

# On-device one-click run (auto-detects CLI or falls back to Python)
scripts/deploy/rk3588_run.sh

# With custom model
scripts/deploy/rk3588_run.sh --model artifacts/models/yolo11n_int8.rknn

# Force Python runner mode
scripts/deploy/rk3588_run.sh --runner python

# Pass-through args to underlying binary
scripts/deploy/rk3588_run.sh -- --json artifacts/out.json

# SSH deployment from PC (when hardware available)
scripts/deploy/deploy_to_board.sh --host <board_ip> --run

# Or with gdbserver for remote debugging
scripts/deploy/deploy_to_board.sh --host <board_ip> --gdb --gdb-port 1234
```

### Performance Optimization & Validation

```bash
# ONNX inference with GPU acceleration (PC validation)
source ~/yolo_env/bin/activate
yolo predict model=artifacts/models/best.onnx source=assets/test.jpg imgsz=640 conf=0.5 iou=0.5 save=true

# PC RKNN simulator (boardless validation)
python scripts/run_rknn_sim.py

# Performance benchmarks
bash scripts/run_bench.sh
```

**Key Performance Findings:**
- **ONNX GPU inference**: 8.6ms (RTX 3060) @ 416×416
- **End-to-end with optimized params**: 16.5ms (60+ FPS) with conf=0.5
- **RKNN PC simulator**: 354ms @ 640×640 (not representative of NPU performance)
- **Expected RK3588 NPU**: 20-40ms @ 640×640 INT8 quantized

**Critical Parameter Tuning:**
- ❌ conf=0.25 (default): 3135ms postprocessing → 0.3 FPS (NMS bottleneck)
- ✅ conf=0.5 (optimized): 5.2ms postprocessing → 60+ FPS (production ready)
- Recommendation: Use conf≥0.5 for industrial applications to avoid excessive false positives

## Graduation Thesis Documentation

**Comprehensive thesis documentation is available in `docs/`**

### Thesis Chapters (Markdown + Word)

The project includes complete graduation thesis documentation:

1. **[Opening Report](docs/thesis_opening_report.md)** (开题报告) ✅
   - Project background and significance
   - Research status and innovation points
   - Technical solution design
   - Timeline planning
   - Exported as `docs/开题报告.docx`

2. **[Chapter 2: System Design](docs/thesis_chapter_system_design.md)** ✅
   - Hardware design (RK3588, dual-NIC configuration)
   - Software architecture (application → system layer)
   - Module design (preprocessing, inference, postprocessing, network)
   - ~3000 words with code examples

3. **[Chapter 3: Model Optimization](docs/thesis_chapter_model_optimization.md)** ✅
   - Model selection and benchmarking (YOLO11n)
   - INT8 quantization methodology
   - Calibration dataset preparation
   - Complete conversion toolchain
   - Resolution optimization (416×416 vs 640×640)
   - ~4000 words with formulas

4. **[Chapter 4: Deployment](docs/thesis_chapter_deployment.md)** ✅
   - Deployment strategy (Python vs C++)
   - Environment setup (PC + board)
   - Complete inference framework code
   - One-click deployment scripts
   - Network integration and serialization
   - ~3500 words with runnable code

5. **[Chapter 5: Performance Testing](docs/thesis_chapter_performance.md)** ✅
   - PC baseline benchmarks (ONNX GPU: 8.6ms)
   - RKNN PC simulator validation
   - Board-level performance projections
   - Parameter tuning impact analysis
   - Graduation requirements compliance
   - ~3500 words with performance tables

**Complete thesis export:** `docs/RK3588行人检测_毕业设计说明书.docx` (69KB, 5 chapters)

**Thesis Statistics:**
- Total chapters: 5 (+ opening report)
- Total word count: ~14,000 words
- Code examples: 20+
- Tables: 30+
- Architecture diagrams: 8+
- Completion: 95% (Phase 4 dataset validation pending)

**Documentation Index:**
See `docs/THESIS_README.md` for complete navigation and usage guide.

**Export to Word:**
```bash
# Using pandoc
pandoc docs/thesis_opening_report.md -o thesis_opening.docx

# All chapters are already exported to .docx format in docs/
ls docs/*.docx
# docs/开题报告.docx
# docs/RK3588行人检测_毕业设计说明书.docx
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
├── .claude/                       # Claude Code automation
│   ├── commands/                  # Slash commands (5 commands)
│   │   ├── full-pipeline.md
│   │   ├── thesis-report.md
│   │   ├── performance-test.md
│   │   ├── board-ready.md
│   │   └── model-validate.md
│   └── skills/                    # Workflow definitions (5 skills)
│       ├── full-pipeline.md
│       ├── thesis-report.md
│       ├── performance-test.md
│       ├── board-ready.md
│       └── model-validate.md
├── docs/                          # Comprehensive documentation
│   ├── thesis_opening_report.md   # 开题报告
│   ├── thesis_chapter_*.md        # 5 thesis chapters
│   ├── *.docx                     # Word exports (2 files)
│   ├── THESIS_README.md           # Documentation index
│   ├── reports/                   # Project status reports
│   ├── deployment/                # Deployment guides
│   └── docs/                      # Technical guides (RGMII, 900Mbps, etc.)
├── apps/                          # Python application (7 modules)
│   ├── config.py                  # Centralized configuration
│   ├── exceptions.py              # Custom exception hierarchy
│   ├── logger.py                  # Unified logging
│   ├── yolov8_rknn_infer.py       # Main RKNN inference app
│   └── utils/
│       ├── preprocessing.py       # Image preprocessing (ONNX/RKNN/board)
│       └── yolo_post.py           # Postprocessing utilities
├── tests/                         # Unit tests (7 files, 40+ cases)
│   └── unit/
│       ├── test_config.py         # 14 tests
│       ├── test_exceptions.py     # 10 tests
│       ├── test_preprocessing.py  # 11 tests
│       └── test_aggregate.py      # 7 tests
├── tools/                         # Core conversion/export tools
│   ├── export_yolov8_to_onnx.py
│   ├── convert_onnx_to_rknn.py
│   ├── aggregate.py, http_receiver.py, http_post.py  # MCP tools
│   └── iperf3_bench.sh, ffprobe_probe.sh
├── scripts/                       # Automation scripts (36 shell scripts)
│   ├── run_bench.sh               # MCP benchmark pipeline
│   ├── run_rknn_sim.py            # PC simulator inference
│   ├── compare_onnx_rknn.py       # Accuracy comparison
│   ├── deploy/
│   │   ├── deploy_to_board.sh     # SSH deployment to RK3588
│   │   └── rk3588_run.sh          # One-click on-device runner
│   ├── benchmark/                 # Performance benchmarks
│   ├── demo/                      # Demo scripts
│   ├── reports/                   # Report generators
│   └── train/                     # Training scripts
├── artifacts/                     # Build outputs and reports
│   ├── models/                    # .onnx and .rknn outputs
│   ├── *_report.md                # Generated reports
│   ├── *.json, *.csv              # Benchmark results
│   └── visualizations/            # Visual comparisons
├── datasets/coco/
│   ├── calib_images/              # Calibration dataset (300 images)
│   └── calib_images/calib.txt     # Absolute paths list
├── config/                        # YAML configs for detection/network
├── configs/mcp_servers.yaml       # MCP server declarations
└── pytest.ini, requirements*.txt  # Configuration files
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

**Virtual env:** `yolo_env` (Python 3.10.12, PyTorch 2.0.1+cu117, CUDA 11.7)

**Key packages (see `requirements.txt`):**
- **Core ML/CV:**
  - numpy>=1.20.0,<2.0 (RKNN toolkit compatibility)
  - opencv-python-headless==4.9.0.80
  - pillow==11.3.0
  - matplotlib==3.10.6

- **YOLO & Training:**
  - ultralytics>=8.0.0 (YOLOv8/v11 training & export)
  - torch>=2.0.0 (for training; omit if inference-only)

- **RKNN Conversion:**
  - rknn-toolkit2>=2.3.2 (ONNX→RKNN conversion on x86 PC)

- **ONNX Inference:**
  - onnxruntime==1.18.1 (PC validation)

- **Configuration:**
  - PyYAML>=6.0

**Development packages (see `requirements-dev.txt`):**
- pytest, pytest-cov (testing with coverage)
- black, pylint, flake8, isort (code quality)
- mypy (type checking)

**Installation:**
```bash
source ~/yolo_env/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

**Activation:**
```bash
source ~/yolo_env/bin/activate
export PYTHONPATH=/home/user/rk-app  # Required for apps/ imports
```

**GPU Support Verification:**
```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Expected: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

**Important Notes:**
- Use numpy<2.0 for RKNN toolkit compatibility
- onnxruntime 1.18.1 works with CUDA 11.7
- For board deployment, only rknn-toolkit2-lite is needed (not full toolkit)

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

## Project Statistics

**Codebase Metrics:**
- **Python modules:** 7 (apps/) + 7 (tests/unit/)
- **Scripts:** 36 shell scripts (scripts/)
- **Test cases:** 40+ unit tests
- **Test coverage:** 88-100% for core modules
- **Documentation:** 35+ markdown files, 2 Word exports
- **Thesis chapters:** 5 chapters + opening report (~14,000 words)
- **Automation:** 5 slash commands + 5 skills

**Model Metrics:**
- **Model size:** 4.7MB (✅ meets <5MB requirement)
- **PC performance:** 8.6ms @ 416×416 (ONNX GPU, RTX 3060)
- **Expected board FPS:** 25-35 FPS (INT8 quantized RKNN)
- **Accuracy:** Mean absolute difference <1% (ONNX vs RKNN)

**Technology Stack:**
- **Languages:** Python 3.10, C++17, Bash
- **Frameworks:** Ultralytics YOLO, RKNN-Toolkit2, ONNX Runtime
- **Build System:** CMake 3.22, pytest
- **Automation:** Claude Code slash commands & skills

## Current Project Status (as of Nov 2025)

### Phase 1 Completed (95%) ✅
- ✅ **Model conversion pipeline** (PyTorch → ONNX → RKNN INT8)
- ✅ **Cross-compilation toolchain** (CMake presets for x86/arm64)
- ✅ **PC boardless validation** (ONNX GPU + RKNN simulator)
- ✅ **One-click deployment script** (`rk3588_run.sh`)
- ✅ **Performance optimization** (conf=0.5 achieves 60+ FPS on PC)
- ✅ **MCP benchmark pipeline** (iperf3 + ffprobe + aggregation)
- ✅ **Unit tests** (40+ test cases, 88-100% coverage)
- ✅ **Code quality** (config, exceptions, logging modules)
- ✅ **Model size** (4.7MB, meets <5MB requirement)
- ✅ **Claude Code automation** (5 slash commands + 5 skills)
- ✅ **Thesis documentation** (5 chapters + opening report, exported to Word)
  - ✅ Opening report (开题报告.docx)
  - ✅ Complete thesis (RK3588行人检测_毕业设计说明书.docx, 69KB)
  - ✅ All chapters with code examples, tables, diagrams

### Phase 2 Pending (Hardware Required) ⏸️

**Dual-NIC Driver Development** (Priority: HIGH)
- ⏸️ Network throughput validation (≥900Mbps)
- ⏸️ Port 1: Industrial camera (1080P capture)
- ⏸️ Port 2: Detection result upload
- 📋 Documentation prepared: `docs/docs/RGMII_NETWORK_GUIDE.md`

**On-Device Performance Testing**
- ⏸️ Actual NPU inference latency measurement
- ⏸️ FPS validation (target: >30 FPS)
- ⏸️ Multi-core NPU parallel processing
- 📋 Deployment scripts ready: `scripts/deploy/rk3588_run.sh`

**Pedestrian Detection Dataset**
- ⏸️ Dataset construction or public dataset selection
- ⏸️ mAP@0.5 validation (target: >90%)
- 📋 Dataset guide prepared: `datasets/PEDESTRIAN_DATASET_GUIDE.md`

### Documentation Status

**Completed:**
- ✅ Opening report (开题报告)
- ✅ 5 thesis chapters (system design, optimization, deployment, performance, etc.)
- ✅ Word exports (.docx format)
- ✅ Technical guides (RGMII, 900Mbps, deployment)
- ✅ Project status reports (compliance, acceptance, honest assessment)

**Pending:**
- ⏸️ Progress report 1: System migration + driver (中期检查1) - awaiting hardware
- ⏸️ Progress report 2: Model deployment (中期检查2) - awaiting Phase 2 data
- ⏸️ Chapter 6: Experimental results - needs board testing data
- ⏸️ Chapter 7: Conclusions - final defense preparation
- ⏸️ English literature translation

### Timeline & Risk Assessment

**Expected Timeline:**
- ✅ **Phase 1 (Oct-Nov 2025):** Thesis + PC validation → 95% complete
- ⏸️ **Phase 2 (Dec 2025):** Dual-NIC driver + system migration
- ⏸️ **Phase 3 (Jan-Apr 2026):** Model deployment + performance tuning
- ⏸️ **Phase 4 (Apr-Jun 2026):** Dataset validation + final thesis
- 📅 **Defense (June 2026)**

**Critical Dependencies:**
- **Hardware availability:** RK3588 board required for Phase 2-4
- **Risk:** If board not available by Dec 2025, Phase 2 milestone will be delayed
- **Mitigation:** All PC-based work completed; can immediately proceed when hardware arrives

**Graduation Requirements Compliance:**
- ✅ Model size <5MB: 4.7MB ✅
- ⏸️ FPS >30: Estimated 25-35 FPS (needs board validation)
- ⏸️ mAP@0.5 >90%: Needs pedestrian dataset validation
- ⏸️ Dual-NIC ≥900Mbps: Needs RGMII driver + testing
- ✅ Working software: PC simulation complete, board deployment scripted
- ✅ Thesis documentation: 5 chapters + opening report complete
