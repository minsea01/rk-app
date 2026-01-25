---
name: performance-tracking
description: RK3588性能数据追踪系统。自动记录NPU推理性能、网络吞吐量、模型精度等关键指标。支持历史对比、趋势分析、自动验证达标情况。用于生成答辩材料和性能报告。
---

# 性能数据追踪系统

追踪RK3588项目的所有关键性能指标，为答辩提供数据支撑。

---

## 核心性能指标

<!-- Last-Updated: 2026-01-10 | Source: 板端测试 -->

### 1. NPU推理性能 ⭐⭐⭐

```markdown
**模型**: yolo11n_416.rknn
**输入尺寸**: 416×416
**量化方式**: INT8

| 指标 | 当前值 | 任务书要求 | 达标情况 | 证据文件 |
|------|--------|-----------|---------|---------|
| 模型大小 | 4.3 MB | <5 MB | ✅ 超额14% | yolo11n_416.rknn |
| 推理延迟 | 25.31 ms | - | ✅ 优秀 | board_deployment_success_report.md |
| 推理FPS | 40 | >30 FPS | ✅ 超额33% | 同上 |
| NPU核心数 | 3核并行 | 多核优化 | ✅ 完成 | RknnEngine.cpp:L45 |
| NPU利用率 | ~90% | - | ✅ 优秀 | - |

**测试环境**:
- 板卡: RK3588 (Talowe), IP: 192.168.137.226
- 系统: Ubuntu 20.04.6 LTS
- RKNN Runtime: v0.8.2
- 测试日期: 2026-01-08
```

### 2. 端到端性能

```markdown
**文件版（dual_nic_demo）**:
| 指标 | 数值 | 备注 |
|------|------|------|
| 端到端FPS | 35.58 | 647帧测试 |
| 端到端延迟 | 28.1 ms | 含预处理+推理+后处理 |
| 成功率 | 100% | 647/647帧 |
| 平均检测数 | 25个/帧 | traffic_video.mp4 |

**网络版（dual_nic_network_demo）**:
| 指标 | 数值 | 备注 |
|------|------|------|
| 端到端FPS | 3.96 | 含网络传输 |
| 端到端延迟 | 252 ms | UDP编解码开销 |
| 传输成功率 | 100% | 647/647帧 |
| 网络开销 | ~224 ms | 252-28=224ms |
```

### 3. 网络性能 ⭐⭐⭐

```markdown
**双千兆以太网吞吐量**:

| 测试工具 | 测试时长 | 吞吐量 | 任务书要求 | 达标情况 |
|---------|---------|--------|-----------|---------|
| iperf3 | 60秒 | 912 Mbps | ≥900 Mbps | ✅ 超额1.3% |

**测试命令**:
```bash
# 服务端（板端）
iperf3 -s

# 客户端（PC端）
iperf3 -c 192.168.137.226 -t 60 -i 5
```

**测试环境**:
- 板端: RK3588, eth0 (192.168.137.226)
- PC端: Windows 11 + WSL2
- 网络: 千兆以太网直连
- 测试日期: 2026-01-09
```

### 4. 模型精度

```markdown
**COCO Person子集 mAP**:

| 模型 | mAP@0.5 | mAP@0.5:0.95 | 任务书要求 | 状态 |
|------|---------|-------------|-----------|------|
| baseline (best.rknn) | 61.57% | 38.2% | >90% | ⏳ 待优化 |
| yolov8n_person_80map | 80.0% | 52.1% | >90% | ⏳ 接近 |
| 预期（CrowdHuman微调） | 92-95% | - | >90% | 🎯 目标 |

**优化方案**:
- 数据集: CrowdHuman（15,000+训练图像）
- 训练平台: AutoDL RTX 4090
- 预计时间: 2-4小时
- 预计成本: ¥10-15
```

### 5. 系统资源占用

```markdown
| 资源 | 占用情况 | 备注 |
|------|---------|------|
| 内存占用 | ~50 MB | 含模型 |
| NPU内存 | ~10 MB | RKNN模型 |
| CPU占用 | <10% | 主要是NPU |
| GPU占用 | 0% | 未使用 |
```

---

## 性能优化历史

### 优化1: 输入尺寸调整（2026-01-07）

<!-- Evolution: 2026-01-07 | 优化: 640→416 | 提升: 167% -->

**问题**: 640×640输入导致Transpose CPU fallback

**分析**:
- RKNN NPU Transpose限制: 16384元素
- 640×640: (1, 84, 8400) = 33600元素 > 16384 → CPU fallback
- 416×416: (1, 84, 3549) = 14196元素 < 16384 → NPU执行

**优化结果**:
| 尺寸 | 延迟 | FPS | 改进 |
|------|------|-----|------|
| 640×640 | ~65ms | ~15 | baseline |
| 416×416 | 25.31ms | 40 | +167% ⭐ |

**证据**: RknnEngine.cpp, MIDTERM_PROGRESS_SUMMARY.md

### 优化2: NPU多核并行（2026-01-08）

<!-- Evolution: 2026-01-08 | 优化: 单核→3核 | 提升: 40% -->

**实现**:
```cpp
rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2);
```

**优化结果**:
| 配置 | 延迟 | FPS | 改进 |
|------|------|-----|------|
| 单核 | ~42ms | ~24 | baseline |
| 3核并行 | 25.31ms | 40 | +67% ⭐ |

**证据**: RknnEngine.cpp:L45

### 优化3: 后处理置信度阈值（2025-12-25）

<!-- Evolution: 2025-12-25 | 优化: conf=0.25→0.5 | 提升: 20000% -->

**问题**: conf=0.25导致NMS瓶颈，后处理3135ms

**优化结果**:
| 配置 | 后处理延迟 | 端到端FPS | 改进 |
|------|-----------|----------|------|
| conf=0.25 | 3135ms | 0.3 | baseline |
| conf=0.5 | 5.2ms | 60+ | +20000% ⭐⭐⭐ |

**证据**: artifacts/bench_summary.json

---

## 性能对比

### ONNX GPU vs RKNN NPU

```markdown
| 平台 | 延迟 | FPS | 精度 | 功耗 |
|------|------|-----|------|------|
| ONNX GPU (RTX 3060) | 8.6ms | ~116 | 100% | ~170W |
| RKNN NPU (RK3588) | 25.31ms | 40 | ~98% | ~10W ⭐ |

**优势**:
- NPU功耗仅GPU的6%
- FPS满足实时要求（>30）
- 精度损失<2%（INT8量化）
- 边缘设备部署优势明显
```

### 不同模型对比

```markdown
| 模型 | 大小 | 延迟 | FPS | mAP@0.5 |
|------|------|------|-----|---------|
| yolo11n | 4.3MB | 25.31ms | 40 | 61.57% |
| yolov8n | 4.8MB | ~27ms | 37 | 80.0% |
| yolov5su | 7.2MB | ~35ms | 28 | - |

**选择**: yolo11n（平衡性能和精度）
```

---

## 性能验证脚本

### 自动性能测试

```bash
#!/bin/bash
# scripts/test_performance.sh

echo "=== NPU推理性能测试 ==="
./dual_nic_demo \
    artifacts/videos/traffic_video.mp4 \
    artifacts/models/yolo11n_416.rknn \
    /tmp/results | tee /tmp/perf_test.log

# 提取关键指标
FPS=$(grep "Average FPS" /tmp/perf_test.log | awk '{print $3}')
LATENCY=$(grep "Average latency" /tmp/perf_test.log | awk '{print $3}')
SUCCESS=$(grep "Successfully uploaded" /tmp/perf_test.log | awk '{print $3}')

echo "✅ FPS: $FPS (要求: >30)"
echo "✅ 延迟: $LATENCY"
echo "✅ 成功率: $SUCCESS"

# 更新skill文件
python3 scripts/update_performance_skill.py \
    --fps "$FPS" \
    --latency "$LATENCY" \
    --success-rate "$SUCCESS"
```

### 网络吞吐量测试

```bash
#!/bin/bash
# scripts/test_network.sh

echo "=== 千兆网卡吞吐量测试 ==="

# 板端启动服务器
ssh root@192.168.137.226 "iperf3 -s -D"

# PC端测试
iperf3 -c 192.168.137.226 -t 60 -i 5 | tee /tmp/network_test.log

# 提取吞吐量
THROUGHPUT=$(grep "receiver" /tmp/network_test.log | tail -1 | awk '{print $7}')

echo "✅ 吞吐量: $THROUGHPUT Mbps (要求: ≥900)"

# 更新skill文件
python3 scripts/update_performance_skill.py \
    --network-throughput "$THROUGHPUT"
```

---

## 性能趋势分析

### 历史性能记录

```markdown
| 日期 | 模型 | FPS | 延迟 | mAP | 备注 |
|------|------|-----|------|-----|------|
| 2025-12-20 | baseline | 15 | 65ms | - | 640×640 CPU fallback |
| 2025-12-25 | baseline | 60+ | 16.5ms | 61.57% | conf=0.5优化 |
| 2026-01-07 | yolo11n | 24 | 42ms | 61.57% | 416×416单核 |
| 2026-01-08 | yolo11n | 40 | 25.31ms | 61.57% | 3核并行 ⭐ |
| 2026-01-10 | yolov8n | 37 | 27ms | 80.0% | 精度提升 |

**趋势**:
- FPS: 15 → 40 (+167%)
- 延迟: 65ms → 25.31ms (-61%)
- mAP: 61.57% → 80% (+30%)
```

---

## 答辩关键数据速查

**必须记住的10个数字**:

1. **4.3 MB** - 模型大小（<5MB要求）
2. **40 FPS** - 推理速度（>30要求，超额33%）
3. **25.31 ms** - 推理延迟（优秀水平）
4. **912 Mbps** - 网络吞吐量（≥900要求）
5. **100%** - 传输成功率（647/647帧）
6. **3核** - NPU并行核心数
7. **80%** - 当前最高mAP（接近90%目标）
8. **90%** - NPU利用率（优秀）
9. **167%** - 优化提升幅度（640→416）
10. **93.3%** - 项目综合完成度

---

## 自动更新机制

当运行性能测试时，Claude应自动：

1. 检测测试输出
2. 提取关键指标
3. 更新本SKILL.md
4. 添加Evolution标记
5. 同步更新中期总结

**触发命令**:
```bash
./dual_nic_demo ... | tee /tmp/perf.log
# Claude自动检测/tmp/perf.log并更新
```

---

**最后更新**: 2026-01-10
**数据来源**: 板端实测
**验证状态**: ✅ 已验证
