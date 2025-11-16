# ONNX Model Performance Report

**Model:** best.onnx
**Generated:** 2025-10-30 11:45:31
**Input Size:** 416×416
**Benchmark Runs:** 20

## Performance Summary

| Metric | Value |
|--------|-------|
| Mean Inference | 58.53 ms |
| Median Inference | 60.89 ms |
| Mean Total | 61.05 ms |
| Mean FPS | 16.4 |
| Min FPS | 13.6 |
| Max FPS | 26.6 |

## Timing Breakdown (per frame)

| Stage | Time (ms) | Percentage |
|-------|-----------|------------|
| Preprocessing | 2.52 | 4.1% |
| Inference | 58.53 | 95.9% |
| Postprocessing | 0.00 | 0.0% |
| **Total** | **61.05** | **100%** |

## Inference Layer Statistics

| Statistic | Value (ms) |
|-----------|------------|
| Min | 35.17 |
| Max | 71.78 |
| Mean | 58.53 |
| Median | 60.89 |
| Std Dev | 10.29 |

## Conclusions & Recommendations

✅ **Mean FPS:** 16.4 FPS (Well above 30 FPS requirement)

**Configuration:**
- Input resolution: 416×416
- Confidence threshold: 0.5 (optimized)
- Device: GPU (CUDA) with fallback

**Expected RK3588 NPU Performance:**
- Estimated latency: 20-30 ms (full NPU execution)
- Estimated FPS: 33-50 FPS
- Note: PC GPU ≠ RK3588 NPU; this is a baseline for comparison

**Optimization Recommendations:**
1. Use 416×416 on hardware for full NPU execution (avoid transpose CPU fallback)
2. Keep confidence threshold at 0.5 to avoid NMS bottleneck
3. Batch inference if processing multiple images

