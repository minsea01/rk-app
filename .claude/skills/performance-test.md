# Performance Test - 综合性能测试套件

Run comprehensive performance benchmarks across all validation environments.

## What this skill does

1. **ONNX GPU Inference**: Test GPU-accelerated inference with onnxruntime-gpu
2. **RKNN PC Simulator**: Test CPU simulator with NHWC format
3. **MCP Benchmark Pipeline**: Network/media/aggregation tests
4. **Parameter Tuning Tests**: Compare different conf/iou threshold combinations
5. **Generate Performance Report**: Consolidated metrics and visualizations

## Parameters

- `models` (optional): List of models to test (default: ["best", "yolo11n"])
- `resolutions` (optional): List of resolutions to test (default: [416, 640])
- `quick` (optional): Run quick tests only (skip extensive parameter sweeps)

## Expected Output

- `artifacts/performance_report_{timestamp}.md` - Comprehensive performance report
- `artifacts/performance_metrics.json` - Machine-readable metrics
- `artifacts/performance_comparison.csv` - CSV format for thesis tables
- Optional: Performance plots/charts

## Usage

Invoke this skill when:
- Validating model performance for thesis
- Comparing different model configurations
- Preparing defense materials
- Optimizing hyperparameters

## Test Suite

### 1. ONNX GPU Inference Test

**What it tests:**
- Inference latency on GPU
- Preprocessing overhead
- Postprocessing (NMS) performance
- End-to-end throughput

**Test variations:**
- Resolutions: 416×416, 640×640
- Confidence thresholds: 0.25, 0.5, 0.7
- Batch sizes: 1 (real-time scenario)

**Expected metrics:**
- Inference time: 8-15ms
- Preprocessing: 2-5ms
- Postprocessing: 5-10ms (with conf=0.5)
- FPS: 50-100

### 2. RKNN PC Simulator Test

**What it tests:**
- RKNN conversion correctness
- PC simulator inference
- Output shape validation
- Memory usage

**Test scenarios:**
- Load ONNX → build → inference
- Different input sizes
- Multi-image batch (if supported)

**Expected metrics:**
- Build time: 10-30s
- Inference time: 200-500ms (CPU simulation, not representative)
- Output correctness: ✅ Shape (1, 84, N)

### 3. MCP Benchmark Pipeline

**What it tests:**
- Network throughput (iperf3)
- Media processing (ffprobe)
- Data aggregation
- HTTP ingestion

**Components:**
1. `iperf3` loopback test → network baseline
2. `ffprobe` 1080p video → media capability
3. Aggregation → data processing
4. HTTP POST → integration validation

**Expected metrics:**
- Network: >30 Gbps (loopback)
- Video: 1080p @ 30fps detected
- Aggregation: <1s
- HTTP: 200 OK

### 4. Parameter Tuning Tests

**Confidence threshold sweep:**
- conf = [0.25, 0.35, 0.45, 0.5, 0.6, 0.7]
- Measure: Postprocessing time, detection count, FPS

**IoU threshold sweep:**
- iou = [0.3, 0.4, 0.45, 0.5, 0.6]
- Measure: NMS effectiveness, final detection count

**Critical finding:**
- ❌ conf=0.25: 3135ms postprocessing (NMS bottleneck)
- ✅ conf=0.5: 5.2ms postprocessing (600× improvement!)

### 5. Comparative Analysis

**ONNX vs RKNN:**
- Model sizes
- Inference speeds
- Output differences (MAE, max error)
- Accuracy metrics (if ground truth available)

**PC vs Expected Board:**
| Metric | PC GPU | PC Simulator | Expected Board |
|--------|--------|--------------|----------------|
| Inference | 8.6ms | 354ms | 30-40ms |
| End-to-End | 16.5ms | N/A | 40-50ms |
| FPS | 60+ | N/A | 25-30 |

## Performance Report Structure

```markdown
# RK3588 Performance Test Report
Date: YYYY-MM-DD
Models Tested: best.onnx, yolo11n.onnx
Test Environment: RTX 3060 Laptop, Ubuntu 22.04

## Executive Summary
- ONNX GPU: 8.6ms inference, 60+ FPS ✅
- RKNN Simulator: 354ms (CPU simulation)
- Optimal parameters: conf=0.5, iou=0.5
- Model size: 4.7MB ✅ (meets <5MB requirement)

## Detailed Results

### 1. ONNX GPU Performance
Resolution: 416×416
- Preprocess: 2.7ms
- Inference: 8.6ms ⚡
- Postprocess: 5.2ms
- **Total: 16.5ms (60.6 FPS)** ✅

Resolution: 640×640
- [Similar breakdown]

### 2. RKNN Simulator
- Build time: 25s
- Inference: 354ms (CPU, not representative)
- Output: (1, 84, 3549) ✅ Correct shape

### 3. MCP Benchmarks
- Network: 32107 Mbps ✅
- Video: 1920×1080 @ 30fps ✅
- Aggregation: Success ✅

### 4. Parameter Tuning
| conf | postprocess | FPS | detections |
|------|-------------|-----|------------|
| 0.25 | 3135ms | 0.3 | many |
| 0.5  | 5.2ms | 60+ | optimal |
| 0.7  | 3.1ms | 70+ | few |

**Recommendation: Use conf=0.5 for production**

## Thesis Metrics Summary

For inclusion in graduation thesis:

✅ Model Size: 4.7MB (Requirement: <5MB)
✅ PC Inference: 8.6ms @ 416×416
✅ PC FPS: 60+ (Requirement: >30)
⏸️ Board FPS: Pending hardware (Expected: 25-30)
⏸️ mAP@0.5: Pending dataset validation (Expected: >90%)

## Conclusions
1. Software layer achieves all performance targets
2. conf=0.5 is critical for real-time performance
3. Board NPU expected to meet >30 FPS requirement
4. Ready for hardware validation
```

## Success Criteria

- ✅ All tests complete without errors
- ✅ Performance metrics documented
- ✅ Thesis-ready tables and charts
- ✅ Optimal parameters identified
- ✅ Board performance predictions provided

## Notes

- GPU tests require onnxruntime-gpu 1.16.3 (CUDA 11.x)
- PC simulator results are not representative of NPU performance
- Board testing requires actual RK3588 hardware
- Some tests may timeout in restricted environments (iperf3)
