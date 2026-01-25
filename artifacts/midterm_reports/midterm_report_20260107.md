# 中期检查报告

**生成时间:** 2026-01-07 14:45
**检查阶段:** 第1阶段 (11-12月)
**项目:** 基于RK3588智能终端的行人检测模块设计

---

## 1. 系统移植

### 验证结果

| 项目 | 状态 | 详情 |
|------|------|------|
| Ubuntu 版本 | ✅ 完成 | Ubuntu 20.04.6 LTS |
| 内核版本 | ✅ 完成 | Linux 5.10.110 aarch64 |
| SSH 连接 | ✅ 完成 | root@192.168.137.226 |
| NPU 驱动 | ✅ 完成 | RKNPU v0.8.2 (DRM) |
| RKNNLite | ✅ 完成 | rknn-toolkit-lite2 2.3.2 |

### 系统信息

```
Host: Talowe EVB-RK3588
System: Ubuntu 20.04.6 LTS
Kernel: Linux 5.10.110 aarch64
CPU: 4×A76 + 4×A55
RAM: 16GB
NPU: 6 TOPS (3 cores)
```

---

## 2. NPU 推理验证

### 测试结果

| 模型 | 输入尺寸 | 推理时间 | FPS | 检测数 | 状态 |
|------|----------|----------|-----|--------|------|
| yolov8n_int8.rknn | 640×640 | 48.69 ms | ~20 | 31 | ✅ 可用 |
| **yolo11n_416.rknn** | **416×416** | **27.52 ms** | **~36** | 25 | ✅ **达标** |

### 性能达标情况

- **目标:** >30 FPS
- **实际:** 36 FPS @ 416×416 ✅
- **三核并行:** core_mask=0x7 (所有3个NPU核心)

### 测试命令

```bash
cd /root/rk-app && \
PYTHONPATH=/root/rk-app python3 apps/yolov8_rknn_infer.py \
    --model artifacts/models/yolo11n_416.rknn \
    --source assets/bus.jpg \
    --save /tmp/npu_result.jpg \
    --imgsz 416 --conf 0.5
```

### 运行日志

```
I RKNN: RKNN Runtime Information, librknnrt version: 2.3.2
I RKNN: RKNN Driver Information, version: 0.8.2
I RKNN: RKNN Model Information, target: RKNPU v2, target platform: rk3588
Inference time: 27.52 ms
Detections: 25
Saved: /tmp/npu_result.jpg
```

---

## 3. 网口驱动配置

### 网卡状态

| 接口 | 状态 | 速度 | 用途 |
|------|------|------|------|
| eth0 | DOWN | - | 相机输入 (待连接) |
| eth1 | UP | 100Mb/s | 检测输出 (当前连接) |

### 驱动信息

```
双千兆网卡已识别:
- eth0: RGMII, MAC b6:4c:e9:3b:97:32
- eth1: RGMII, MAC ba:4c:e9:3b:97:32

当前连接: eth1 @ 192.168.137.226/24
```

### 待办事项

- [ ] 更换千兆网线 (当前为百兆)
- [ ] 连接 eth0 到 GigE 相机
- [ ] 吞吐量验证 (≥900Mbps)

---

## 4. 网络吞吐量验证

### 测试状态

- **目标:** ≥900 Mbps
- **当前状态:** 待测试 (需千兆网线)
- **工具:** iperf3 已安装

### 预计验证命令

```bash
# 板子端启动服务器
iperf3 -s

# PC端测试
iperf3 -c 192.168.137.226 -t 10
```

---

## 5. 进度对照

| 任务项 | 计划时间 | 状态 | 备注 |
|--------|----------|------|------|
| 系统移植 | 11-12月 | ✅ 完成 | Ubuntu 20.04.6 LTS |
| NPU 驱动 | 11-12月 | ✅ 完成 | RKNPU v0.8.2 |
| NPU 推理 | 11-12月 | ✅ 完成 | 36 FPS @ 416×416 |
| 网口驱动 | 11-12月 | ✅ 完成 | 双网卡识别正常 |
| 吞吐量验证 | 11-12月 | ⏳ 待验证 | 需千兆网线 |
| 第一阶段报告 | 12月 | ✅ 本报告 | |

---

## 6. 关键指标达成

| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| 模型大小 | <5MB | 4.5MB (yolo11n_416.rknn) | ✅ |
| 推理帧率 | >30 FPS | 36 FPS | ✅ |
| 网络吞吐 | ≥900Mbps | 待测试 | ⏳ |
| mAP精度 | ≥90% | 61.57% (待微调) | ⏳ |

---

## 7. 佐证材料

- 检测结果图片: `artifacts/npu_result_bus.jpg`
- 部署包: `rk-app-board-deploy.tar.gz` (57MB)
- 模型文件: `artifacts/models/*.rknn`
- 推理脚本: `apps/yolov8_rknn_infer.py`

---

## 8. 下一阶段计划

1. **硬件准备:**
   - 更换千兆网线，验证 ≥900Mbps 吞吐量
   - 连接 GigE 工业相机到 eth0

2. **模型优化:**
   - CityPersons 数据集微调，提升 mAP 到 ≥90%
   - 云端 AutoDL 4090 训练

3. **系统集成:**
   - 端到端延迟优化 (<45ms)
   - UDP 推流功能开发

---

*报告由 Claude Code 自动生成*
*项目: rk-app | 答辩: 2026年6月*
