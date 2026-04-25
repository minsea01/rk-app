# Detection Configs

当前建议只用下面三类配置，避免误跑旧模型路径。

| 场景 | 配置 | 说明 |
| --- | --- | --- |
| 海康 MV-CA020-20GC | `config/detection/detect_hikrobot_mv_ca020_20gc.yaml` | eth0 接相机，eth1 `192.168.137.56` 上传到 PC `192.168.137.1:9000` |
| 真实 GigE 工业相机默认入口 | `config/detection/detect_rknn.yaml` | 当前同样指向 `MV-CA020-20GC`，便于演示时少输配置名 |
| 双网口验收 | `config/detection/detect_gige_dual_nic.yaml` | 与主线相同，保留双网口命名便于报告引用 |
| 无相机假输入 | `config/detection/detect_fake_camera.yaml` | 使用预生成 `artifacts/fake_camera_416_30fps_60s.avi` 模拟 30fps 输入 |

相机到货后先在板端跑：

```bash
cd /root/rk-app-new
scripts/deploy/prepare_hikrobot_gige.sh --apply-network
scripts/deploy/prepare_hikrobot_gige.sh --expect-camera --grab 30
```

如果 `arv-tool` 发现的相机名称不是 `MV-CA020-20GC`，把 YAML 中
`source.uri` 的 `camera-name=` 替换成发现到的精确名称，或者临时用
`detect_cli --source "camera-name=实际名称,width=1920,height=1200,framerate=30/1,format=BGR"`。

旧的 ONNX/COCO/host 测试配置已经统一改到当前存在的
`best_person_aug_416_norm.onnx` 或 `best_person_aug_416_norm_int8.rknn`，
不再引用缺失的 `best.onnx`、`pedestrian_416.rknn` 或旧 ONNX/RKNN 文件名。
