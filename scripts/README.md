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
  - `generate_achievement_report.py`
  - `charts/gen_chart.py`
  - `charts/gen_code_charts.py`
  - `charts/gen_more_charts.py`
- `benchmark/`, `demo/`, `datasets/`, `train/`, `tune/`, `network/`, `maintenance/`: scenario-specific automation

Root-level ad hoc scripts have been folded into these subdirectories so the
repository root only keeps build, packaging, dependency, and repository
metadata files.
