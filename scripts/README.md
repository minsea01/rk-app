Scripts layout

- `deploy/`: Board deployment and runtime helpers
  - `deploy_to_board.sh`
  - `rk3588_run.sh`
  - `sync_sysroot.sh`
- `evaluation/`: Offline validation and debugging entrypoints
  - `batch_inference.py`
  - `official_yolo_map.py`
  - `pedestrian_map_evaluator.py`
- `profiling/`: Performance profiling and latency measurement
  - `performance_profiler.py`
  - `end_to_end_latency.py`
- `reports/`: Report generation assets
  - `charts/gen_chart.py`
  - `charts/gen_code_charts.py`
  - `charts/gen_more_charts.py`
- Live result viewers:
  - `live_viewer.py`: OpenCV window/headless latest-frame receiver for embedded JPEG streams
  - `live_web_viewer.py`: Browser-based receiver for WSL or headless hosts
  - `demo/start_live_web_demo.sh`: one-command WSL live demo startup
  - `demo/status_live_web_demo.sh`, `demo/stop_live_web_demo.sh`: inspect/stop the demo
- `benchmark/`, `demo/`, `datasets/`, `train/`, `tune/`, `network/`, `maintenance/`: scenario-specific automation

Root-level ad hoc scripts have been folded into these subdirectories so the
repository root only keeps build, packaging, dependency, and repository
metadata files.
