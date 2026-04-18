# rk-app（RK3588 工业目标检测流水线）

简洁、模块化的 RK3588 目标检测工程：C++ 实现采集→预处理→推理（ONNX/RKNN）→后处理→输出全链路；配套 C++ CLI 与板端 Python Runner，训练/导出/压测工具链完整可复现。

本仓库遵循 `AGENTS.md` 的目录与命名规范（src/include/examples/apps/tools/scripts/artifacts/configs/docker 等）。

## 快速开始
- 本机构建（x86 调试）
  - `cmake --preset x86-debug && cmake --build --preset x86-debug`
  - 运行 ONNX 管线：`./build/x86-debug/detect_cli --cfg config/detection/detect.yaml --json output/detections.json --save_vis output/vis`
- RK3588 交叉构建（启用 RKNN）
  - `cmake --preset arm64-release -DENABLE_RKNN=ON && cmake --build --preset arm64`
  - 安装产物：`cmake --install build/arm64`
- 板端 Python Runner（RKNN 推理）
  - `python apps/yolov8_rknn_infer.py --model artifacts/models/best.rknn --names config/classes.txt --source assets/test.jpg --save artifacts/vis.jpg`

更多演示与环境变量示例见 `docs/guides/QUICK_START_GUIDE.md` 与 `docs/QUICKSTART.md`。

## detect_cli 配置（阶段 3）

`detect_cli` 现在只接受结构化运行时 schema，不再兼容早期的平铺/别名字段。最小可用配置如下：

```yaml
source:
  type: video
  uri: assets/street.mp4

engine:
  type: onnx
  model: artifacts/models/yolo11n_opset12_416.onnx
  input_size: [416, 416]

postprocess:
  conf_threshold: 0.25
  nms_threshold: 0.45
  max_detections: 100

classes:
  path: config/classes.txt

output:
  type: tcp
  tcp:
    host: 127.0.0.1
    port: 9000
    queue_size: 32

runtime:
  warmup: 5
  async: false

logging:
  level: INFO
```

说明：
- `output.type` 当前以 `tcp` 为唯一正式输出通道；离线 JSON 和可视化通过 `--json`、`--save_vis` CLI 参数导出。
- 模型解码契约只认模型同名 sidecar：`<model>.json` 或 `<model>.meta`。
- 已移除的旧键包括：`input`、`nms`、`engine.imgsz`、`output.ip`、`output.tcp.queue`、`perf`、根级 `log_level`。

## Python 依赖分层
- 基础运行时（轻量）：
  - `pip install -r requirements.txt`
- 训练/导出/ONNX 工具链（重依赖）：
  - `pip install -r requirements_train.txt`
- 板端运行（在基础运行时上追加 RKNN Lite）：
  - `pip install -r requirements_board.txt`

## 训练与模型导出
- 一键训练+导出（Ultralytics+ONNX→RKNN）：
  - `make RUN_NAME=<exp> all MODEL_PREFIX=yolo11n`
- 手动导出 INT8 RKNN：
  - `python tools/convert_onnx_to_rknn.py --onnx artifacts/models/yolo11n.onnx --out artifacts/models/yolo11n_int8.rknn --calib datasets/calib`
- PC/RKNN 结果对齐与基准：
  - `python tools/pc_compare.py ...`，`scripts/run_bench.sh` 聚合 iperf3/ffprobe 指标到 `artifacts/bench_summary.*`

模型与命名约定见 `artifacts/models/README.md`（例如：`<model>[_variant].onnx` 与 `_int8.rknn`）。

## 测试与基准
- 单元测试（GoogleTest）：`ctest --preset x86-debug`
- 网络/多媒体基准：`scripts/run_bench.sh`（输出位于 `artifacts/`）

## 目录结构
- `src/`：核心流水线实现（`capture/`、`preprocess/`、`infer/`、`post/`、`output/`）
- `include/`：公共头文件（遵循 `include/rkapp` 的布局与命名）
- `examples/`：示例 CLI（`detect_cli.cpp`）
- `apps/`：板端 Python Runner（`yolov8_rknn_infer.py` 等）
- `tools/`：训练、导出、评估与聚合脚本（如 `convert_onnx_to_rknn.py`、`model_evaluation.py`）
- `scripts/`：运行/部署/压测等封装脚本（如 `run_bench.sh`）
- `artifacts/`：模型与基准产出（`artifacts/models/*`、`bench_summary.*`）
- `datasets/`：数据集与校准集（不纳入版本控制的体量数据）
- `configs/`：共享实验预设与工具链锁定
- `docker/`：可复现实验环境
- `docs/`：深度技术文档与交付材料

## 构建开关与依赖
- `-DENABLE_ONNX=ON`（默认）：主机侧 ONNXRuntime 推理；可通过 `ORT_HOME` 指向外部 ORT。
- `-DENABLE_RKNN=ON`：开启 RKNN SDK（默认路径 `RKNN_HOME=/opt/rknpu2`）。
- `-DENABLE_GIGE=ON`：启用基于 GStreamer 的 GigE 源（需要 `aravis/gstreamer`）。

## 常见命令速查
- 主机调试：`cmake --preset x86-debug && cmake --build --preset x86-debug`
- 交叉编译：`cmake --build --preset arm64 -DENABLE_RKNN=ON`
- 运行 CLI：`./build/x86-debug/detect_cli --cfg config/detection/detect.yaml`
- 运行板端：`python apps/yolov8_rknn_infer.py --model artifacts/models/best.rknn`
- 板端一键跑：`scripts/deploy/rk3588_run.sh`（自动设置库路径并运行 CLI，缺少二进制时回退 Python Runner）
- 单元测试：`ctest --preset x86-debug`
- 压测汇总：`scripts/run_bench.sh`

## 参考文档
- 快速开始：`docs/guides/QUICK_START_GUIDE.md`
- 板端依赖：`docs/guides/BOARD_RUNTIME_DEPENDENCIES.md`
- 硬件集成：`docs/guides/HARDWARE_INTEGRATION_MANUAL.md`
- 部署清单：`docs/RK3588_VALIDATION_CHECKLIST.md`
- 性能分析：`docs/PERFORMANCE_ANALYSIS.md`
- 项目报告：`docs/reports/` (状态报告、成果总结等)
- 毕业论文：`docs/thesis/` (开题报告、论文章节等)
- 代码与提交规范：`AGENTS.md`
