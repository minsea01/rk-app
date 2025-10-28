Run comprehensive performance benchmarks across all validation environments.

## Task

Execute complete performance test suite:

1. ONNX GPU inference test (with onnxruntime-gpu)
2. RKNN PC simulator test
3. MCP benchmark pipeline (network/media/aggregation)
4. Parameter tuning tests (conf/iou threshold comparison)
5. Generate consolidated performance report

## Test Parameters

Ask user (or use defaults):
- Models to test: ["best", "yolo11n"] (default)
- Resolutions: [416, 640] (default)
- Quick mode: false (set true to skip extensive parameter sweeps)

## Expected Tests

### 1. ONNX GPU Test
```bash
source ~/yolo_env/bin/activate
export PYTHONPATH=/home/minsea/rk-app
yolo predict model=artifacts/models/best.onnx source=assets/test.jpg imgsz=640 conf=0.5 iou=0.5 save=true
```

### 2. RKNN Simulator Test
```bash
python scripts/run_rknn_sim.py
```

### 3. MCP Benchmark
```bash
bash scripts/run_bench.sh
```

### 4. Parameter Sweep
Test conf thresholds: [0.25, 0.5, 0.7]
Measure: preprocessing, inference, postprocessing time

## Expected Metrics

- Inference latency (ms)
- End-to-end throughput (FPS)
- Preprocessing overhead
- Postprocessing (NMS) time
- Model size
- Network throughput
- Media processing capability

## Critical Finding to Document

⚠️ **conf=0.25 vs conf=0.5 Performance Impact:**
- conf=0.25: 3135ms postprocessing (NMS bottleneck)
- conf=0.5: 5.2ms postprocessing (600× improvement!)

## Output Files

- `artifacts/performance_report_{timestamp}.md` - Main report
- `artifacts/performance_metrics.json` - Machine-readable data
- `artifacts/performance_comparison.csv` - Thesis table format

## Success Criteria

- All tests complete without errors
- Performance metrics documented
- Thesis-ready tables and comparisons
- Optimal parameters identified
- Board performance predictions provided
