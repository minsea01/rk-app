# Detection Configs

当前建议只用下面三类配置，避免误跑旧模型路径。

| 场景 | 配置 | 说明 |
| --- | --- | --- |
| 海康 MV-CA020-20GC | `config/detection/detect_hikrobot_mv_ca020_20gc.yaml` | eth1 接相机，eth0 `192.168.137.226` 上传到 PC `192.168.137.1:9000` |
| 真实 GigE 工业相机默认入口 | `config/detection/detect_rknn.yaml` | 当前自动选择第一台 Aravis 相机，默认 `GRAY8` |
| 双网口验收 | `config/detection/detect_gige_dual_nic.yaml` | 与主线相同，保留双网口命名便于报告引用 |
| 无相机假输入 | `config/detection/detect_fake_camera.yaml` | 使用预生成 `artifacts/fake_camera_416_30fps_60s.avi` 模拟 30fps 输入 |

相机到货后先在板端跑：

```bash
cd /opt/rk_app_current
scripts/deploy/prepare_hikrobot_gige.sh --apply-network
scripts/deploy/prepare_hikrobot_gige.sh --expect-camera --grab 30
```

如果现场需要指定精确相机名，在 YAML 的 `source.uri` 前面加上
`camera-name=<实际 Aravis 名称>,`。默认不写 `camera-name` 会自动选择第一台
GigE Vision 相机。当前板端实测默认格式是 `GRAY8`；若换成彩色或 Bayer
输出，再把 `format=` 改成对应格式。

现场调相机曝光/增益时，也可以把 Aravis 源属性直接写进 `source.uri`：

```yaml
source:
  type: gige
  uri: "width=1920,height=1200,framerate=30/1,format=GRAY8,exposure-auto=false,exposure=8000,gain-auto=false,gain=6,pull-timeout-ms=1000,max-failures=10"
```

`exposure` 单位是微秒。提高帧率时优先关掉自动曝光并把曝光时间降到
`8000` 到 `12000`，画面变暗再加补光或少量提高 `gain`。

旧的 ONNX/COCO/host 测试配置已经统一改到当前存在的
`best_person_aug_416_norm.onnx` 或 `best_person_aug_416_norm_int8.rknn`，
不再引用缺失的 `best.onnx`、`pedestrian_416.rknn` 或旧 ONNX/RKNN 文件名。
