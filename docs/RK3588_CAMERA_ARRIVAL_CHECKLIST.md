# 工业相机到货后检查单

## 网络

- 保持 eth1：`192.168.137.56/24`，用于 SSH 和结果上传到 PC `192.168.137.1:9000`。
- 配置 eth0 为相机网段，例如 `192.168.1.10/24`，不要给 eth0 配默认网关。
- 检查链路：

```bash
ip -brief addr
ethtool eth0
ethtool eth1
ip route
```

## 相机发现和取流

```bash
arv-tool-0.6 list || arv-tool-0.8 list || arv-tool-0.10 list
gst-inspect-1.0 aravissrc
```

先跑板端准备脚本：

```bash
cd /root/rk-app-new
scripts/deploy/prepare_hikrobot_gige.sh --apply-network
scripts/deploy/prepare_hikrobot_gige.sh --expect-camera --grab 30
```

当前项目已经准备了海康 `MV-CA020-20GC` 专用配置：

```text
config/detection/detect_hikrobot_mv_ca020_20gc.yaml
```

其默认取流参数为：

```yaml
source:
  type: gige
  uri: "camera-name=MV-CA020-20GC,width=1920,height=1200,framerate=30/1,format=BGR,pull-timeout-ms=500,max-failures=10"
```

如果 `arv-tool` 列出的名称不是 `MV-CA020-20GC`，按实际名称替换
`camera-name=`。如果 `format=BGR` 协商失败，先用 Hikrobot MVS 查看
PixelFormat，再尝试 `BayerRG8`、`BayerGB8`、`BayerBG8`、`BayerGR8`
或 `Mono8`。

## 真实闭环测试

PC 端：

```bash
# 保存 JSON/JPEG 证据
python3 scripts/results_receiver.py --host 192.168.137.1 --port 9000 \
  --output-dir artifacts/received_results --no-health

# 或实时显示检测画面；WSLg/桌面环境下会弹出窗口，按 q 退出
python3 scripts/live_viewer.py --host 192.168.137.1 --port 9000 \
  --save-latest artifacts/live_view/latest.jpg
```

板端：

```bash
cd /root/rk-app-new
timeout 30s ./build/board/detect_cli \
  --cfg config/detection/detect_hikrobot_mv_ca020_20gc.yaml \
  --json artifacts/gige_camera_30s_eth1.json \
  2>&1 | tee artifacts/gige_camera_30s_eth1.log
```

验收提取：

```bash
grep -E "Frames processed|Frames dropped|Average FPS|Average latency" artifacts/gige_camera_30s_eth1.log
grep -c "RGA imresize failed\\|fallback to CPU" artifacts/gige_camera_30s_eth1.log
ip -s link show eth0
ip -s link show eth1
```

## 通过标准

| 项目 | 标准 |
| --- | --- |
| 相机发现 | `arv-tool` 能列出相机 |
| 取流 | `detect_cli` 连续运行 30 秒不中断 |
| 帧率 | 平均 FPS >= 30 |
| 延迟 | 平均延迟 <= 45 ms，优先补 P95 |
| 丢帧 | `Frames dropped = 0` 或明确低于验收阈值 |
| RGA | `RGA imresize failed` 为 0，或说明 fallback 原因 |
| 网络 | eth0/eth1 error/drop/CRC/collision 为 0 |
