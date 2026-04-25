# Environment Requirements and Testing Guide

This document clarifies which scripts can run on PC vs. RK3588 board, and provides a testing guide for boardless development.

## 📋 Environment Matrix

### PC Environment (Development & Testing)

**Operating System:** WSL2 Ubuntu 22.04 / Native Linux

**Python Version:** 3.8+

**Required Packages:**
```bash
# Install all dependencies
pip install -r requirements-dev.txt

# Core packages
- numpy>=1.20.0,<2.0
- opencv-python-headless==4.9.0.80
- ultralytics>=8.0.0
- rknn-toolkit2>=2.3.2    # PC-only, for ONNX→RKNN conversion
- onnxruntime==1.18.1
```

### RK3588 Board Environment (Deployment Only)

**Operating System:** Linux ARM64

**Python Version:** 3.8+

**Required Packages:**
```bash
# On RK3588 board only
pip install rknn-toolkit-lite2>=2.3.2  # Board runtime library
pip install opencv-python
pip install numpy
```

---

## 🔧 Script Compatibility Table

| Script/Tool | PC | RK3588 | Dependencies | Purpose |
|-------------|:--:|:------:|--------------|---------|
| **Model Preparation** |
| `tools/export_yolov8_to_onnx.py` | ✅ | ❌ | ultralytics | Export PyTorch to ONNX |
| `tools/convert_onnx_to_rknn.py` | ✅ | ❌ | rknn-toolkit2 | Convert ONNX to RKNN |
| **PC Validation** |
| `scripts/run_rknn_sim.py` | ✅ | ❌ | rknn-toolkit2 | PC simulator inference |
| `scripts/compare_onnx_rknn.py` | ✅ | ❌ | rknn-toolkit2, onnxruntime | Accuracy comparison |
| `tools/compare.py` | ✅ | ❌ | onnxruntime, numpy | Model output comparison |
| **Board Deployment** |
| `apps/yolov8_rknn_infer.py` | ❌ | ✅ | rknnlite | On-device inference |
| `apps/yolov8_stream.py` | ❌ | ✅ | rknnlite | Video stream inference |
| `scripts/deploy/deploy_to_board.sh` | ✅ | N/A | ssh, scp | Deploy to board via SSH |
| **Testing & Utilities** |
| `tests/unit/*.py` | ✅ | ✅ | pytest | Unit tests |
| `tools/aggregate.py` | ✅ | ✅ | - | Benchmark aggregation |
| `scripts/run_bench.sh` | ✅ | ✅ | iperf3, ffprobe | Benchmark pipeline |

**Legend:**
- ✅ Can run
- ❌ Cannot run
- N/A Not applicable

---

## 🚀 Boardless Development Workflow

### 1. Setup Environment (One-time)

```bash
# Activate virtual environment (if using)
source ~/yolo_env/bin/activate

# Install dependencies
pip install -r requirements-dev.txt

# Verify installation
python3 -c "from apps.config import ModelConfig; print('✓ Core modules OK')"
python3 -c "import onnxruntime; print('✓ ONNX Runtime OK')"
python3 -c "from rknn.api import RKNN; print('✓ RKNN Toolkit2 OK')"
```

### 2. Model Export & Conversion

```bash
# Step 1: Export YOLOv8/v11 to ONNX
python3 tools/export_yolov8_to_onnx.py \
  --weights yolo11n.pt \
  --imgsz 416 \
  --outdir artifacts/models

# Step 2: Convert ONNX to RKNN
python3 tools/convert_onnx_to_rknn.py \
  --onnx artifacts/models/best_person_aug_416_norm.onnx \
  --out artifacts/models/best_person_aug_416_norm_int8.rknn \
  --calib datasets/coco/calib_images/calib.txt \
  --target rk3588 \
  --do-quant
```

### 3. PC Simulator Validation

```bash
# Test RKNN inference on PC (no hardware needed)
python3 scripts/run_rknn_sim.py \
  --model artifacts/models/best_person_aug_416_norm.onnx \
  --image assets/test.jpg \
  --imgsz 416

# Test with custom image
python3 scripts/run_rknn_sim.py \
  --image datasets/coco/calib_images/000000002261.jpg
```

### 4. Accuracy Verification

```bash
# Compare ONNX vs RKNN inference outputs
python3 scripts/compare_onnx_rknn.py

# Results saved to: artifacts/onnx_rknn_comparison.json
```

### 5. Run Unit Tests

```bash
# Run all unit tests
pytest tests/unit -v

# Run with coverage report
pytest tests/unit -v --cov=apps --cov=tools --cov-report=html

# Open coverage report
# firefox htmlcov/index.html
```

---

## 🐛 Common Issues & Solutions

### Issue: `ModuleNotFoundError: No module named 'cv2'`

**Solution:**
```bash
pip install opencv-python-headless==4.9.0.80
# or
pip install -r requirements.txt
```

### Issue: `ModuleNotFoundError: No module named 'rknn'`

**Solution:**
```bash
pip install rknn-toolkit2>=2.3.2
```

**Note:** This is the PC version. Do NOT install `rknn-toolkit-lite2` on PC.

### Issue: Calibration file paths are wrong

**Symptom:** RKNN conversion fails with "image path not found"

**Solution:**
```bash
# Manually regenerate
cd datasets/coco/calib_images
find $(pwd) -name "*.jpg" | sort > calib.txt
```

### Issue: `ImportError: rknnlite not found` when running apps/yolov8_rknn_infer.py

**Expected behavior:** This is normal on PC. This script requires RK3588 hardware.

**Solution:** Use `scripts/run_rknn_sim.py` instead for PC testing.

---

## 📊 What Can Be Validated Without Board

### ✅ Fully Testable on PC

- [x] Model export (PyTorch → ONNX)
- [x] Model conversion (ONNX → RKNN)
- [x] RKNN inference simulation (PC simulator)
- [x] Accuracy comparison (ONNX vs RKNN)
- [x] Unit tests (config, preprocessing, exceptions)
- [x] Code quality (linting, formatting)
- [x] Preprocessing pipeline
- [x] Post-processing (NMS, decoding)
- [x] Benchmark utilities

### ⚠️ Partially Testable

- [~] Performance benchmarks (simulated, not actual NPU)
- [~] Calibration dataset preparation (can prepare, but not validate quantization on real NPU)

### ❌ Requires RK3588 Hardware

- [ ] Actual NPU performance (latency, throughput)
- [ ] rknnlite runtime testing
- [ ] On-device camera capture
- [ ] Network streaming (dual-NIC)
- [ ] Industrial camera (GigE Vision) integration
- [ ] Real-world deployment validation

---

## 🎯 Pre-deployment Checklist (Boardless)

Before deploying to RK3588, ensure all these pass on PC:

```bash
# 1. Core imports work
python3 -c "from apps.config import ModelConfig; print('OK')"
python3 -c "from apps.exceptions import PreprocessError; print('OK')"
python3 -c "from apps.utils.preprocessing import preprocess_onnx; print('OK')"

# 2. Unit tests pass
pytest tests/unit -v

# 3. Model export works
python3 tools/export_yolov8_to_onnx.py --weights yolo11n.pt --imgsz 416

# 4. ONNX validation succeeds
python3 tools/onnx_bench.py --model artifacts/models/best_person_aug_416_norm.onnx

# 5. RKNN conversion completes
python3 tools/convert_onnx_to_rknn.py --onnx artifacts/models/best_person_aug_416_norm.onnx --calib datasets/coco/calib_images/calib.txt

# 6. PC simulator runs
python3 scripts/run_rknn_sim.py

# 7. Accuracy comparison acceptable
python3 scripts/compare_onnx_rknn.py
# Check: mean_abs_diff < 0.02 (2%)
```

---

## 📚 Additional Resources

- **README.md** - Complete project guide and workflow
- **QUICK_START_GUIDE.md** - Quick start commands
- **RK3588_VALIDATION_CHECKLIST.md** - On-device validation steps
- **README.md** - Project overview

---

## 💡 Tips for Efficient Boardless Development

1. **Use PC simulator extensively** - Catch issues before hardware deployment
2. **Compare ONNX vs RKNN outputs** - Validate quantization accuracy early
3. **Test with multiple images** - Ensure robustness across different inputs
4. **Monitor memory usage** - Use `top` or `htop` during conversion
5. **Keep calibration dataset diverse** - 300+ images from target domain
6. **Version control models** - Tag ONNX and RKNN models together
7. **Document changes** - Note any model modifications in git commits

---

**Last Updated:** 2025-10-27
**Maintainer:** rk-app development team
