# 🐳 RK3588 Docker部署完整指南

**优势**: 环境隔离、依赖管理简单、一键部署、易于迁移

---

## 📦 两种部署方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **Docker部署** | ✅ 环境隔离<br>✅ 依赖打包<br>✅ 一键部署<br>✅ 易于迁移 | ❌ 镜像体积大<br>❌ NPU访问复杂<br>❌ 需要Docker支持 | 板子已有Docker<br>多环境部署 |
| **原生部署** | ✅ 性能最优<br>✅ NPU直接访问<br>✅ 镜像体积小 | ❌ 依赖手动安装<br>❌ 环境配置复杂 | 单板运行<br>追求极致性能 |

**推荐策略**:
- **PC模拟阶段**: 使用`docker-compose.dual-nic.yml`进行网络流水线测试
- **板上部署阶段**: 优先使用**原生部署**（性能最优），Docker作为备选方案

---

## 🎯 方案一：Docker部署到板子（适合已有Docker的板子）

### 前提条件

```bash
# 在RK3588板子上检查Docker
ssh radxa@<板子IP>

# 检查Docker是否安装
docker --version
# 应该显示: Docker version 20.10.x+

# 检查Docker Compose
docker-compose --version
# 应该显示: docker-compose version 1.29.x+

# 如果没有安装Docker，参考官方文档:
# https://docs.docker.com/engine/install/ubuntu/
```

### Step 1: 在WSL上构建ARM64镜像（5分钟）

```bash
cd /home/user/rk-app

# 使用自动化部署脚本
bash scripts/deploy/docker_deploy.sh <板子IP>

# 例如:
bash scripts/deploy/docker_deploy.sh 192.168.1.100
```

**脚本会自动完成:**
1. 构建ARM64镜像（使用buildx）
2. 保存镜像为tar文件
3. SCP传输到板子
4. 在板子上加载镜像
5. 启动容器

### Step 2: 验证容器运行（板子上）

```bash
# SSH到板子
ssh radxa@192.168.1.100

# 查看容器状态
docker ps

# 查看日志
docker logs rk3588-detector

# 进入容器交互
docker exec -it rk3588-detector bash
```

### Step 3: 运行推理（容器内）

```bash
# 在容器内执行
python3 apps/yolov8_rknn_infer.py \
  --model /app/artifacts/models/yolo11n_416.rknn \
  --source /app/assets/test.jpg \
  --save /app/logs/result.jpg
```

---

## 🌐 方案二：Docker双网卡仿真（PC上测试双网口流水线）

**用途**: 在PC上模拟RK3588双网卡环境，测试完整流水线

### 架构

```
┌─────────────────────────────────────────────────────────┐
│  docker-compose.dual-nic.yml                            │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────┐      ┌──────────────────────┐      │
│  │ camera_server   │      │  rk3588_simulator    │      │
│  │ 192.168.1.101   │─────▶│  192.168.1.100 (eth0)│      │
│  │ (相机模拟器)      │      │  192.168.2.100 (eth1)│─────┐│
│  └─────────────────┘      └──────────────────────┘     ││
│         ▲                                               ││
│         │ camera_network (192.168.1.0/24)              ││
│                                                          ││
│                    detection_network (192.168.2.0/24)   ││
│         │                                               ││
│         ▼                          ┌──────────────────┐││
│  ┌─────────────────┐               │ results_server   │││
│  │ network_monitor │               │ 192.168.2.101    ││◀┘
│  │ (可选)           │               │ (结果接收器)      ││
│  └─────────────────┘               └──────────────────┘│
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Step 1: 启动完整仿真环境

```bash
cd /home/user/rk-app

# 启动所有服务（相机模拟器 + 检测器 + 结果接收器）
docker-compose -f docker-compose.dual-nic.yml up -d

# 查看服务状态
docker-compose -f docker-compose.dual-nic.yml ps

# 查看日志
docker-compose -f docker-compose.dual-nic.yml logs -f

# 启动网络监控（可选）
docker-compose -f docker-compose.dual-nic.yml --profile monitor up -d
```

### Step 2: 验证网络连通性

```bash
# 进入RK3588模拟器容器
docker exec -it rk3588_detection bash

# 检查网络接口
ip addr show

# Ping相机网络
ping -c 3 192.168.1.101

# Ping结果服务器
ping -c 3 192.168.2.101

# 测试网络吞吐量
# 在results_server容器中运行iperf3 server
docker exec -d results_server iperf3 -s

# 在rk3588_detection容器中测试
docker exec rk3588_detection iperf3 -c 192.168.2.101 -t 10
```

### Step 3: 监控网络流量（可选）

```bash
# 进入监控容器
docker exec -it network_monitor bash

# 抓包分析
tcpdump -i eth0 -w /pcap/camera_traffic.pcap

# 实时查看流量
iftop -i eth0
```

### Step 4: 停止仿真环境

```bash
docker-compose -f docker-compose.dual-nic.yml down
```

---

## 🔧 Docker配置详解

### Dockerfile.rk3588 关键配置

```dockerfile
FROM arm64v8/ubuntu:20.04  # ARM64基础镜像

# 安装Python和OpenCV
RUN apt-get update && apt-get install -y \
    python3 python3-pip libopencv-dev python3-opencv

# 复制项目文件
COPY apps/ /app/apps/
COPY artifacts/models/ /app/artifacts/models/

# 设置环境变量
ENV PYTHONPATH=/app
```

### docker-compose.rk3588.yml 关键配置

```yaml
services:
  rk3588-detector:
    network_mode: host        # 使用主机网络（直接访问eth0/eth1）
    privileged: true          # 特权模式（NPU访问）

    devices:
      - /dev/dri:/dev/dri     # GPU/NPU设备

    volumes:
      - /opt/rknpu2:/opt/rknpu2:ro  # 挂载RKNN运行时库

    environment:
      - LD_LIBRARY_PATH=/opt/rknpu2/lib
```

**重要说明**:
- `network_mode: host` - 容器直接使用主机网络，可直接访问eth0/eth1
- `privileged: true` - 允许访问NPU设备（可能需要，取决于板子配置）
- 挂载`/opt/rknpu2` - 使用主机的RKNN运行时库

---

## 📊 性能对比测试

### Docker vs 原生性能对比

```bash
# 1. 在Docker容器中测试
docker exec rk3588-detector \
  python3 scripts/profiling/board_benchmark.py \
  --model /app/artifacts/models/yolo11n_416.rknn \
  --iterations 100 \
  --output /app/artifacts/docker_performance.json

# 2. 在主机上测试（原生）
python3 scripts/profiling/board_benchmark.py \
  --model artifacts/models/yolo11n_416.rknn \
  --iterations 100 \
  --output artifacts/native_performance.json

# 3. 比较结果
python3 << EOF
import json

with open('artifacts/docker_performance.json') as f:
    docker_perf = json.load(f)
with open('artifacts/native_performance.json') as f:
    native_perf = json.load(f)

print(f"Docker FPS:  {docker_perf['fps']['mean']:.2f}")
print(f"Native FPS:  {native_perf['fps']['mean']:.2f}")
print(f"Overhead:    {(native_perf['fps']['mean'] - docker_perf['fps']['mean']) / native_perf['fps']['mean'] * 100:.1f}%")
EOF
```

**预期结果**:
- Docker FPS: 35-40 FPS（与原生接近，因为使用host网络）
- Native FPS: 40-45 FPS
- Overhead: <10%（可接受的开销）

---

## ✅ Docker部署毕设验证清单

### 功能验证
- [ ] 镜像成功构建（ARM64架构）
- [ ] 容器成功启动（docker ps显示running）
- [ ] NPU设备可访问（/dev/dri, /dev/rknpu0）
- [ ] 推理正常运行（容器内测试通过）

### 性能验证
- [ ] **FPS ≥ 30**: Docker容器内FPS测试
- [ ] **延迟 ≤ 45ms**: 端到端延迟测试
- [ ] **开销 < 10%**: Docker vs 原生性能对比

### 网络验证（使用docker-compose.dual-nic.yml）
- [ ] 双网络连通性（192.168.1.x ↔ 192.168.2.x）
- [ ] 网络吞吐量 ≥ 900Mbps（iperf3测试）
- [ ] 相机流接收（camera_server → rk3588_simulator）
- [ ] 结果上传（rk3588_simulator → results_server）

---

## 🆚 决策建议

### 使用Docker部署的场景
✅ 板子已经安装Docker
✅ 需要部署多个环境
✅ 需要环境隔离（多个应用共存）
✅ 开发阶段需要频繁迭代

### 使用原生部署的场景
✅ **追求极致性能**（推荐）
✅ 板子资源有限（Docker额外占用内存）
✅ NPU驱动配置复杂（原生更容易调试）
✅ **答辩演示**（减少Docker依赖，降低失败风险）

**最终建议**:
- **毕设答辩**: 使用**原生部署**（性能最优，演示更稳定）
- **日常开发**: 可使用Docker仿真（快速迭代）
- **生产环境**: 取决于实际需求

---

## 🔥 常见问题

### Docker中NPU不可用

**问题**: `Error: cannot access /dev/rknpu0`

**解决**:
```bash
# 1. 确认主机NPU可用
ls -l /dev/rknpu*

# 2. 在docker-compose.yml中添加设备映射
devices:
  - /dev/rknpu0:/dev/rknpu0

# 3. 使用privileged模式
privileged: true
```

### ARM64镜像构建失败

**问题**: `ERROR: failed to solve: platform not supported`

**解决**:
```bash
# 启用Docker buildx
docker buildx create --use --name multiarch

# 构建时指定平台
docker buildx build --platform linux/arm64 -t rk3588-detector .
```

### 容器内网络不通

**问题**: 容器内无法访问主机网口

**解决**:
```yaml
# 使用主机网络模式（docker-compose.yml）
network_mode: host

# 或使用macvlan（高级）
networks:
  eth0_network:
    driver: macvlan
    driver_opts:
      parent: eth0
```

---

## 📚 参考命令速查

```bash
# 【PC端】构建并部署Docker
cd /home/user/rk-app
bash scripts/deploy/docker_deploy.sh 192.168.1.100

# 【PC端】启动双网卡仿真
docker-compose -f docker-compose.dual-nic.yml up -d

# 【板子端】查看容器状态
docker ps
docker logs rk3588-detector

# 【板子端】进入容器
docker exec -it rk3588-detector bash

# 【板子端】容器内推理
docker exec rk3588-detector \
  python3 apps/yolov8_rknn_infer.py \
  --model /app/artifacts/models/yolo11n_416.rknn \
  --source /app/assets/test.jpg

# 【板子端】停止容器
docker-compose -f docker-compose.rk3588.yml down
```

---

## 🚀 快速决策流程图

```
板子是否已有Docker？
    │
    ├─ YES ─▶ 是否追求极致性能？
    │           │
    │           ├─ YES ─▶ 使用原生部署 ✅ (推荐)
    │           │          参考: docs/guides/BOARD_QUICKSTART.md
    │           │
    │           └─ NO ──▶ 使用Docker部署
    │                     参考: 本文档
    │
    └─ NO ──▶ 使用原生部署 ✅ (推荐)
              参考: docs/guides/BOARD_QUICKSTART.md
```

**最终建议**: 对于毕设答辩，**原生部署**是最佳选择！性能最优，演示最稳定。

---

**总结**: Docker部署适合多环境、快速迭代场景；原生部署适合追求性能和稳定性场景。你的项目已经两种方案都准备好了！🎉
