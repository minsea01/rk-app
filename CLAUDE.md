# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RK3588 industrial edge AI system for real-time pedestrian detection with dual-NIC network streaming. Graduation design for North University of China.

- **Platform:** Rockchip RK3588 (6 TOPS NPU, 3 cores, 4xA76+4xA55, 16GB RAM)
- **Model:** YOLOv8/YOLO11 -> ONNX -> RKNN INT8 quantized
- **Dev environment:** WSL2 Ubuntu 22.04, Python venv `yolo_env`
- **C++ standard:** C++17, Google style (`.clang-format`), 2-space indent, 100 cols
- **Python:** 3.10, black (100 cols), isort, flake8, 4-space indent

## Build Commands

```bash
# Host debug (ONNX, with ASAN/UBSAN)
cmake --preset x86-debug && cmake --build --preset x86-debug

# Host release (recommended for development)
cmake --preset native-release && cmake --build --preset native-release

# RK3588 cross-compile (RKNN)
cmake --preset arm64-release -DENABLE_RKNN=ON && cmake --build --preset arm64

# Run CLI
./build/x86-debug/detect_cli --cfg config/detection/detect.yaml
```

**CMake presets:** `x86-debug` (sanitizers on), `x86-debug-nosan`, `native-release`, `arm64-release`

**CMake options:**
- `-DENABLE_ONNX=ON` (default) - host ONNX inference; `ORT_HOME` for external path
- `-DENABLE_RKNN=ON` - RKNN SDK (default `RKNN_HOME=/opt/rknpu2`)
- `-DENABLE_GIGE=ON` - GigE camera via aravis/gstreamer
- `-DENABLE_MPP=ON` - hardware video decode (RK3588)
- `-DENABLE_RGA=ON` - hardware preprocessing (RK3588)

ARM64 builds use `-march=armv8.2-a+crypto+fp16 -mtune=cortex-a76` tuning. Sanitizers auto-enable on x86 Debug unless using `x86-debug-nosan`.

## Test Commands

```bash
# C++ unit tests (GoogleTest, auto-fetched)
ctest --preset x86-debug

# Python unit tests (skip hardware-dependent)
pytest tests/unit -m "not requires_hardware" -v

# Single Python test file
pytest tests/unit/test_config_loader.py -v

# Python tests with coverage
pytest tests/unit -v --cov=apps --cov=tools --cov-report=html

# Full sanity check (C++ + Python)
cmake --preset x86-debug && cmake --build --preset x86-debug && ctest --preset x86-debug
pytest tests/unit -m "not requires_hardware" -v
```

**Pytest markers:** `unit`, `integration`, `slow`, `requires_hardware`, `requires_model`

## Lint and Format

```bash
# All pre-commit hooks at once
pre-commit run --all-files

# Individual tools
black apps/ tools/ tests/
isort apps/ tools/ tests/
flake8 apps/ tools/ tests/
```

Pre-commit hooks: black, clang-format (Google style), isort, flake8, bandit, shellcheck, trailing-whitespace, end-of-file-fixer, check-yaml, check-json, markdownlint, check-added-large-files (5MB).

## Python Environment

```bash
source ~/yolo_env/bin/activate
export PYTHONPATH=$PWD  # Required for apps/ imports
```

Key constraint: `numpy<2.0` (RKNN toolkit compatibility).

## C++ Architecture

**Interface hierarchy (all use factory functions):**
```
ISource (capture)    -> FolderSource, VideoSource, GigeSource, MppSource
IInferEngine (infer) -> OnnxEngine, RknnEngine
IOutput (output)     -> TcpOutput, UdpOutput
```

**High-level API (Pimpl pattern):**
```cpp
DetectionPipeline pipeline;
pipeline.init(config);
while (auto result = pipeline.next()) { /* process detections */ }
```

**Compiled libraries:**
- `rkapp_core` - capture/preprocess/postprocess/output
- `rkapp_infer_onnx` - ONNX inference engine (supports external ORT via `ORT_HOME`)
- `rkapp_infer_rknn` - RKNN NPU engine (multi-core scheduling, DMA-BUF zero-copy)
- `rkapp_pipeline` - high-level DetectionPipeline (sync + async modes)
- `rkapp_decode_utils` - shared YOLO decode (DFL + RAW), platform-independent

**Design patterns:** RAII (smart pointers 95%+), Pimpl, factory functions (`createSource()`, `createEngine()`).

### Zero-Copy Hardware Pipeline (RK3588)

The critical performance path avoids all CPU memory copies:

```
[MPP decode] → DMA-BUF fd → [RGA resize/pad] → DMA-BUF fd → [NPU infer]
   NV12           (zero-copy)     RGB888            (zero-copy)     uint8
```

- `DmaBuf` (`include/rkapp/common/DmaBuf.hpp`) - DRM GEM/CMA allocation, fd sharing, cache sync primitives (`syncForCpu{Read,Write}{Start,End}`)
- `FramePool` - pre-allocated frame pool (avoids runtime allocation)
- RGA preprocessing: 0.3ms vs OpenCV 3ms (10x speedup)
- MPP hardware decode: 50% CPU reduction

### RknnEngine Thread Safety

`RknnEngine` is fully thread-safe for multi-threaded pipelines:
- `state_mutex_` protects all member variable access (getters/setters)
- `impl_->infer_mutex` serializes RKNN context calls during inference
- `impl_->shutting_down` atomic flag for graceful shutdown
- `impl_` is `shared_ptr` (safe capture in async callbacks)
- NPU core mask: default 0x7 = `RKNN_NPU_CORE_0_1_2` (tri-core, 6 TOPS)

## Python Conventions

- **Config:** Use `apps/config_loader.py` (priority: CLI > ENV > YAML > defaults). Dataclasses in `apps/config.py`.
- **Exceptions:** Use custom hierarchy in `apps/exceptions.py` (`PreprocessError`, `InferenceError`, etc.), not bare `except`.
- **Logging:** Use `apps/logger.py`, not `print()`.
- **Decode metadata:** `apps/utils/decode_meta.py` mirrors C++ metadata logic.

## Model Conversion Workflow

```bash
# 1. PyTorch -> ONNX (opset 12, simplify=True)
python3 tools/export_yolov8_to_onnx.py --weights yolo11n.pt --imgsz 640 --outdir artifacts/models

# 2. ONNX -> RKNN INT8
python3 tools/convert_onnx_to_rknn.py \
  --onnx artifacts/models/yolo11n.onnx \
  --out artifacts/models/yolo11n.rknn \
  --calib datasets/coco/calib_images/calib.txt \
  --target rk3588 --do-quant

# 3. PC simulator validation
python3 scripts/run_rknn_sim.py

# 4. ONNX vs RKNN accuracy comparison
python3 scripts/compare_onnx_rknn.py
```

**Makefile shortcuts:** `make all`, `make compare`, `make calib`, `make validate`, `make convert-fp16`

## Decode Metadata

For ambiguous YOLO output channels, the engine requires sidecar metadata. Lookup order:
1. `<model>.json` or `<model>.meta` (next to model file)
2. `artifacts/models/decode_meta.json` (global fallback)

**DFL head** (YOLOv8/YOLO11 native):
```json
{"head": "dfl", "reg_max": 16, "strides": [8, 16, 32], "num_classes": 80}
```

**RAW head** (YOLOv5-style):
```json
{"head": "raw", "num_classes": 80, "has_objectness": true}
```

Conservative policy: fails fast if metadata missing or ambiguous (prevents silent decode errors). `reg_max` capped at 32 (`kMaxSupportedRegMax`).

## Critical Pitfalls

**RKNN Transpose CPU fallback:** NPU has 16384-element limit for Transpose ops.
- 640x640 -> (1,84,8400) = 33600 elements -> falls back to CPU
- 416x416 -> (1,84,3549) = 14196 elements -> stays on NPU
- Use 416x416 for production.

**Calibration paths must be absolute:** `convert_onnx_to_rknn.py` fails silently with relative paths. Generate with:
```bash
find calib_images -name "*.jpg" -exec realpath {} \; > calib_images/calib.txt
```

**PC simulator vs board runtime:**
- PC Simulator (rknn-toolkit2): `rk.load_onnx()` + `rk.build()`, NHWC input `(1,H,W,3)`, call `rk.config()` before `rk.load_onnx()`
- Board (rknn-toolkit2-lite): loads `.rknn` directly, expects uint8 (0-255)

**Data format conventions:**
- ONNX Runtime: NCHW `(1,3,H,W)`
- RKNN PC Simulator: NHWC `(1,H,W,3)`
- Preprocessing: BGR to RGB via `img[..., ::-1]`, keep uint8 for RKNN

**NMS bottleneck:** conf=0.25 causes 3135ms postprocessing (0.3 FPS). Use conf>=0.5 for production (5.2ms, 60+ FPS).

## CI Pipeline

GitHub Actions (`.github/workflows/ci.yml`) triggers on push to `main/master/develop/claude/**` and PRs. Jobs: python-quality (black+flake8), python-tests, file-validation (shellcheck), cpp-build-tests (x86-debug + ctest), model-validation, docs-check. The `ci-success` gate uses `if: always()` and warns but does not hard-fail.

## Git Conventions

- Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, `test:`
- **No `Co-Authored-By` tags** - this is a solo graduation project
- Keep commits small and focused

## Config and Directory Notes

- `config/` is the primary config directory; `configs/` is a symlink for compatibility
- Config priority: CLI > ENV > YAML > defaults (via `apps/config_loader.py`)
- Model artifacts go in `artifacts/models/` with naming: `<model>[_variant].onnx`, `<model>_int8.rknn`
- Datasets/calibration in `datasets/` (not committed to git)

## Slash Commands

- `/full-pipeline` - PyTorch -> ONNX -> RKNN -> Validation
- `/thesis-report` - graduation thesis progress report
- `/performance-test` - ONNX GPU, RKNN sim benchmarks
- `/board-ready` - RK3588 deployment readiness check
- `/model-validate` - ONNX vs RKNN accuracy comparison
