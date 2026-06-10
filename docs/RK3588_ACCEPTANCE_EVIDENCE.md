# RK3588 Acceptance Evidence

更新时间：2026-04-24

## 当前结论

图表：`artifacts/visualizations/rk3588_acceptance_dashboard.png`，
`artifacts/visualizations/rk3588_latency_evidence.png`。

| 项目 | 实测结果 | 判定 | 证据 |
| --- | ---: | --- | --- |
| 单网口 eth1 上行 | 942 Mbit/s，0 retransmits | 通过 | iperf3，板子 `192.168.137.56` -> PC `192.168.137.1` |
| 单网口 eth1 下行 | 941 Mbit/s，0 retransmits | 通过 | iperf3 reverse |
| 单网口 eth1 双向 | 约 897-899 Mbit/s each way，0 retransmits | 通过 | iperf3 `--bidir` |
| NPU 单模型基准 | 18.75 ms，53.35 FPS | 通过 | `/root/rk-app-new/artifacts/bench_npu_core7_acceptance_20260424.json` |
| C++ 假相机闭环 | 1800/1800 帧，0 丢帧，28.924 ms，33.96 FPS | 通过 | `/root/rk-app-new/artifacts/fake_camera_acceptance_config_20260424.log` |
| TCP 上传到 PC | 1800 行 JSON，约 1.35 MB | 通过 | PC 接收端日志，板子 JSON 文件 |
| RGA stride fallback | 0 次 | 通过 | `grep "RGA imresize failed\\|fallback to CPU"` |
| eth1 网口错误 | errors/dropped/CRC/collision = 0 | 通过 | `ip -s link show eth1` / `ethtool -S eth1` |

## 当前可复现实测命令

板端 NPU 单模型基准：

```bash
cd /root/rk-app-new
python3 scripts/profiling/board_benchmark.py \
  --model artifacts/models/best_person_aug_416_norm_int8.rknn \
  --imgsz 416 --warmup 3 --iterations 30 --core-mask 0x7 \
  --output artifacts/bench_npu_core7_acceptance_20260424.json
```

生成无相机假视频：

```bash
cd /root/rk-app-new
python3 tools/make_fake_camera_video.py \
  --source artifacts/pexels_random15_416 \
  --output artifacts/fake_camera_416_30fps_60s.avi \
  --fps 30 --frames 1800 --width 416 --height 416
```

PC 端启动接收端：

```bash
python3 scripts/results_receiver.py --host 192.168.137.1 --port 9000 \
  --output-dir artifacts/received_results --no-health
```

板端跑假相机闭环：

```bash
cd /root/rk-app-new
./build/board/detect_cli \
  --cfg config/detection/detect_fake_camera.yaml \
  --json artifacts/fake_camera_acceptance_config_20260424.json
```

## 未完成项

真实工业相机还未接入，原因是相机电源未到。真实相机到货后只需要补：

- eth0 相机网段配置和相机发现
- GigE 取流格式、帧率、曝光稳定性
- 30 秒和 5 分钟真实采集闭环
- 真实输入下的 RGA fallback 次数和 P95 延迟
