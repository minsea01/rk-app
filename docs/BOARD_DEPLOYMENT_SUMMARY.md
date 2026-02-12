# RK3588板子部署资源汇总

**生成时间**: 2025-11-21
**目的**: 汇总从WSL到RK3588板子部署所需的所有资源

---

## 🎯 部署方案选择

你的项目**已经有两套完整的部署方案**：

| 方案 | 优点 | 缺点 | 适用场景 | 文档 |
|------|------|------|----------|------|
| **原生部署** ✅ | 性能最优<br>NPU直接访问<br>部署包小 | 依赖手动安装 | **毕设答辩**<br>追求极致性能 | `docs/guides/BOARD_QUICKSTART.md` |
| **Docker部署** | 环境隔离<br>依赖打包<br>一键部署 | 镜像体积大<br>NPU配置复杂 | 多环境部署<br>快速迭代 | `DOCKER_DEPLOYMENT_GUIDE.md` |

**推荐**: 对于毕设答辩，使用**原生部署**（性能最优，演示最稳定）

---

## 📚 新增文档（共5个）

### 1. 快速入门指南
**文件**: `docs/guides/BOARD_QUICKSTART.md`
**用途**: 4步骤、30分钟完成首次推理的快速指南（原生部署）
**关键内容**:
- WSL打包传输流程
- 板子环境配置
- 首次推理验证
- 性能测试命令
- 常见问题排查

### 2. 完整部署清单
**文件**: `docs/deployment/RK3588_DEPLOYMENT_CHECKLIST.md`
**用途**: 详细的11步部署流程和毕设验证清单
**关键内容**:
- 硬件准备清单
- 系统镜像烧录
- 网络连接配置
- 依赖安装详解
- 双网卡配置验证
- 答辩材料收集指南
- 时间规划表（4-5天完整验证）

### 3. 快速部署指南（已有）
**文件**: `docs/deployment/BOARD_DEPLOYMENT_QUICKSTART.md`
**用途**: 30分钟快速部署指南
**关键内容**:
- 5步骤快速部署
- 常见问题处理
- 性能验证方法
- 毕设验证清单

### 4. Docker部署指南
**文件**: `docs/deployment/DOCKER_DEPLOYMENT_GUIDE.md`
**用途**: 完整的Docker部署方案（含双网卡仿真）
**关键内容**:
- Docker vs 原生部署对比
- 方案一：Docker部署到板子
- 方案二：Docker双网卡仿真（PC测试）
- 性能对比测试
- 决策建议

### 5. 本汇总文档
**文件**: `docs/BOARD_DEPLOYMENT_SUMMARY.md`
**用途**: 所有部署资源的索引

---

## 🔧 新增脚本（共4个）

### 1. 打包脚本
**文件**: `scripts/deploy/pack_for_board.sh`
**用途**: 将WSL项目打包成板上部署包
**功能**:
- 自动打包Python代码、RKNN模型、脚本、配置
- 排除不必要文件（__pycache__, .pyc）
- 生成 `rk-app-board-deploy.tar.gz` (~30MB)
- 包含板上README和精简requirements

**使用方法**:
```bash
cd /home/user/rk-app
bash scripts/deploy/pack_for_board.sh
# 输出: rk-app-board-deploy.tar.gz

# 传输到板子
scp rk-app-board-deploy.tar.gz radxa@<IP>:/home/radxa/
```

### 2. 板上性能测试脚本
**文件**: `scripts/profiling/board_benchmark.py`
**用途**: RK3588板上FPS和延迟基准测试
**功能**:
- 测量推理延迟（mean, median, min, max, p95, p99）
- 计算FPS和吞吐量
- 自动验证毕设要求（FPS≥30, 延迟≤45ms）
- 输出JSON报告
- 支持warmup和多次迭代

**使用方法**:
```bash
# 在板子上执行
python3 scripts/profiling/board_benchmark.py \
  --model artifacts/models/yolo11n_416.rknn \
  --iterations 100 \
  --imgsz 416 \
  --output artifacts/board_performance.json

# 输出示例:
# Mean latency: 22.5ms
# Mean FPS: 44.4
# ✅ FPS ≥ 30: PASS
# ✅ Latency ≤ 45ms: PASS
```

**关键参数**:
- `--iterations`: 测试迭代次数（默认100）
- `--warmup`: 预热迭代次数（默认10）
- `--core-mask`: NPU核心掩码（默认0x7，使用3个核心）
- `--verbose`: 显示详细进度

### 3. 端到端延迟测试脚本
**文件**: `scripts/profiling/end_to_end_latency.py`
**用途**: 测量完整流水线延迟（预处理+推理+后处理+网络）
**功能**:
- 分阶段测量延迟：
  - Preprocessing（图像读取+预处理）
  - Inference（NPU推理）
  - Postprocessing（解码+NMS）
  - Network TX（TCP/UDP传输，可选）
- 统计每个阶段的mean/median/min/max/p95/p99
- 验证总延迟是否≤45ms

**使用方法**:
```bash
# 在板子上执行
python3 scripts/profiling/end_to_end_latency.py \
  --model artifacts/models/yolo11n_416.rknn \
  --source assets/test.jpg \
  --imgsz 416 \
  --iterations 100 \
  --output artifacts/e2e_latency.json

# 带网络传输测试
python3 scripts/profiling/end_to_end_latency.py \
  --model artifacts/models/yolo11n_416.rknn \
  --source assets/test.jpg \
  --target-host 192.168.2.200 \
  --target-port 8080 \
  --iterations 100
```

**输出示例**:
```
Preprocessing:  3.2ms
Inference:      22.5ms
Postprocessing: 5.1ms
Network TX:     8.5ms
====================
Total:          39.3ms

Graduation Requirement (≤45ms): ✅ PASS
```

### 4. 双网口流水线脚本
**文件**: `scripts/deploy/dual_nic_pipeline.sh`
**用途**: 实现相机输入→推理→结果上传的完整双网口流水线
**功能**:
- 从eth0（192.168.1.x）接收相机RTSP流
- RKNN推理处理
- 通过eth1（192.168.2.x）上传检测结果
- 支持JSON/UDP输出格式
- 自动验证网口配置

**使用方法**:
```bash
# 在板子上执行

# 方式A: RTSP相机流
bash scripts/deploy/dual_nic_pipeline.sh \
  --input-source rtsp://192.168.1.100:8554/stream \
  --output-host 192.168.2.200 \
  --output-port 8080 \
  --model artifacts/models/yolo11n_416.rknn

# 方式B: USB摄像头（/dev/video0）
bash scripts/deploy/dual_nic_pipeline.sh \
  --input-source 0 \
  --output-host 192.168.2.200 \
  --model artifacts/models/yolo11n_416.rknn

# 方式C: 单张图片测试
bash scripts/deploy/dual_nic_pipeline.sh \
  --input-source assets/test.jpg \
  --output-host 192.168.2.200

# UDP低延迟输出
bash scripts/deploy/dual_nic_pipeline.sh \
  --input-source rtsp://192.168.1.100:8554/stream \
  --output-host 192.168.2.200 \
  --format udp
```

**关键参数**:
- `--input-interface eth0`: 输入网口（相机连接）
- `--input-source <url>`: 相机流URL（RTSP/HTTP/USB设备号）
- `--output-interface eth1`: 输出网口（服务器连接）
- `--output-host <ip>`: 目标服务器IP（必需）
- `--format json|udp`: 输出格式（JSON或UDP）

---

## 📦 已有脚本（共8个）

### 原生部署脚本（5个）

#### 1. 一键运行脚本
**文件**: `scripts/deploy/rk3588_run.sh`
**用途**: 板上一键运行推理（自动选择CLI或Python）

#### 2. SSH部署脚本
**文件**: `scripts/deploy/deploy_to_board.sh`
**用途**: 从PC通过SSH远程部署到板子

#### 3. 依赖安装脚本
**文件**: `scripts/deploy/install_dependencies.sh`
**用途**: 自动安装板上所有Python依赖

#### 4. 健康检查脚本
**文件**: `scripts/deploy/board_health_check.sh`
**用途**: 7层健康检查（Python、依赖、NPU、网络等）

#### 5. 双网卡配置脚本
**文件**: `scripts/deploy/configure_dual_nic.sh`
**用途**: 配置RGMII双千兆网卡（eth0+eth1）

### Docker部署脚本（3个）

#### 6. Docker自动化部署脚本
**文件**: `scripts/deploy/docker_deploy.sh`
**用途**: 自动构建ARM64镜像并部署到板子
**功能**:
- 构建ARM64镜像（使用buildx）
- 保存镜像为tar
- SCP传输到板子
- 在板子上加载并启动容器

**使用方法**:
```bash
bash scripts/deploy/docker_deploy.sh 192.168.1.100
```

#### 7. Dockerfile.rk3588
**文件**: `Dockerfile.rk3588`
**用途**: RK3588 ARM64镜像定义
**包含**: Python环境、依赖、应用代码、RKNN模型

#### 8. docker-compose配置（2个）
- **docker-compose.rk3588.yml**: 板上单容器部署配置
  - `network_mode: host` - 直接访问主机网络（eth0/eth1）
  - `privileged: true` - 访问NPU设备
  - 挂载RKNN运行时库

- **docker-compose.dual-nic.yml**: PC双网卡仿真配置
  - 相机模拟器（192.168.1.101）
  - RK3588模拟器（双网口: 192.168.1.100 + 192.168.2.100）
  - 结果接收器（192.168.2.101）
  - 网络监控（可选）

---

## 🎯 完整部署流程

### 方案A: 原生部署（推荐，4步，30分钟）

#### 第一步：WSL打包（5分钟）
```bash
cd /home/user/rk-app
bash scripts/deploy/pack_for_board.sh
scp rk-app-board-deploy.tar.gz radxa@<IP>:/home/radxa/
```

#### 第二步：板子解压检查（5分钟）
```bash
ssh radxa@<IP>
tar xzf rk-app-board-deploy.tar.gz
cd rk-app
bash scripts/deploy/board_health_check.sh
```

#### 第三步：安装依赖（10分钟）
```bash
bash scripts/deploy/install_dependencies.sh
```

#### 第四步：首次推理（5分钟）
```bash
export PYTHONPATH=$PWD
python3 apps/yolov8_rknn_infer.py \
  --model artifacts/models/yolo11n_416.rknn \
  --source assets/test.jpg \
  --save result.jpg
```

### 方案B: Docker部署（需要板子已有Docker）

#### 一键部署（10分钟）
```bash
cd /home/user/rk-app

# 自动构建+传输+部署
bash scripts/deploy/docker_deploy.sh 192.168.1.100
```

#### 验证运行
```bash
# SSH到板子
ssh radxa@192.168.1.100

# 查看容器
docker ps

# 查看日志
docker logs rk3588-detector

# 容器内推理
docker exec rk3588-detector \
  python3 apps/yolov8_rknn_infer.py \
  --model /app/artifacts/models/yolo11n_416.rknn \
  --source /app/assets/test.jpg
```

### 方案C: Docker双网卡仿真（PC上测试流水线）

```bash
cd /home/user/rk-app

# 启动完整仿真环境（相机+检测器+结果接收器）
docker-compose -f docker-compose.dual-nic.yml up -d

# 查看日志
docker-compose -f docker-compose.dual-nic.yml logs -f

# 停止
docker-compose -f docker-compose.dual-nic.yml down
```

---

## 📊 性能测试流程

### 基准FPS测试
```bash
python3 scripts/profiling/board_benchmark.py \
  --model artifacts/models/yolo11n_416.rknn \
  --iterations 100
```

### 端到端延迟测试
```bash
python3 scripts/profiling/end_to_end_latency.py \
  --model artifacts/models/yolo11n_416.rknn \
  --source assets/test.jpg \
  --iterations 100
```

### 双网卡吞吐量测试
```bash
sudo bash scripts/deploy/configure_dual_nic.sh

# 在另一台PC运行: iperf3 -s
iperf3 -c <server_ip> -B 192.168.1.100 -t 10 -P 4
iperf3 -c <server_ip> -B 192.168.2.100 -t 10 -P 4
```

### 双网口流水线测试
```bash
bash scripts/deploy/dual_nic_pipeline.sh \
  --input-source rtsp://192.168.1.100:8554/stream \
  --output-host 192.168.2.200
```

---

## ✅ 毕设验证清单

### 功能指标
- [ ] 模型成功加载（RKNNLite初始化返回0）
- [ ] 推理正常运行（单张图片测试通过）
- [ ] 检测结果正确（输出bbox、置信度、类别）

### 性能指标（核心要求）
- [ ] **模型体积 ≤ 5MB**: yolo11n_416.rknn = 4.3MB ✅
- [ ] **FPS ≥ 30**: 目标35-45 FPS @ 416×416
- [ ] **延迟 ≤ 45ms**: 端到端延迟测试
- [ ] **内存占用 ≤ 500MB**: 峰值内存监控

### 网络指标（双网卡要求）
- [ ] **双网口识别**: eth0 + eth1 都能up
- [ ] **吞吐量 ≥ 900Mbps**: iperf3测试每个网口
- [ ] **端口1接收**: 能从相机网络接收1080P流
- [ ] **端口2上传**: 能上传检测结果到服务器

### 行人检测（准确率要求）
- [ ] **mAP@0.5 ≥ 90%**: 需要CityPersons微调
  - 当前基线: 61.57% (YOLO11n预训练)
  - 微调路径: `bash scripts/train/train_citypersons.sh`（2-4小时）

---

## 🔍 关键文件位置

### 模型文件
```
artifacts/models/
├── yolo11n_416.rknn     (4.3MB, 推荐，避免CPU回退)
├── yolo11n_int8.rknn    (4.7MB)
└── best.rknn            (4.7MB)
```

### 配置文件
```
config/
├── detection/
│   └── detect_rknn.yaml
└── industrial_classes.txt
```

### 测试图片
```
assets/
├── test.jpg
└── (其他测试图片)
```

### 输出目录
```
artifacts/
├── models/              (RKNN模型)
├── board_performance.json
├── e2e_latency.json
└── result.jpg           (推理输出)
```

---

## 🆘 常见问题快速索引

| 问题 | 解决方法 | 文档位置 |
|------|---------|---------|
| NPU初始化失败 | `sudo modprobe rknpu` | docs/guides/BOARD_QUICKSTART.md |
| rknn-toolkit-lite2安装失败 | 手动下载wheel | RK3588_DEPLOYMENT_CHECKLIST.md |
| 网口不识别 | 检查dmesg和设备树 | docs/guides/BOARD_QUICKSTART.md |
| 推理速度慢 | 使用416×416模型 | docs/guides/BOARD_QUICKSTART.md |
| OpenCV导入错误 | 安装libgl1-mesa-glx | docs/guides/BOARD_QUICKSTART.md |

---

## 📅 时间规划

| 阶段 | 时间 | 任务 |
|------|------|------|
| 硬件准备 | 1天 | 购买板子、烧录镜像 |
| 基础部署 | 0.5天 | 打包、传输、安装依赖 |
| 功能验证 | 0.5天 | 首次推理、性能测试 |
| 网络配置 | 1天 | 双网卡配置、吞吐量测试 |
| 流水线集成 | 1天 | 相机接入、结果上传 |
| mAP微调（可选） | 2-4小时 | CityPersons微调（在PC） |
| 答辩材料 | 1天 | 截图、拍照、报告 |
| **总计** | **4-5天** | **完整验证** |

---

## 📚 参考资料

### 项目文档
- **快速入门**: `docs/guides/BOARD_QUICKSTART.md`
- **完整清单**: `docs/deployment/RK3588_DEPLOYMENT_CHECKLIST.md`
- **部署指南**: `docs/deployment/BOARD_DEPLOYMENT_QUICKSTART.md`
- **RGMII网络**: `docs/docs/RGMII_NETWORK_GUIDE.md`
- **项目说明**: `CLAUDE.md`

### 外部资源
- **Rockchip RKNN**: https://github.com/rockchip-linux/rknn-toolkit2
- **Radxa Wiki**: https://wiki.radxa.com/Rock5
- **Orange Pi**: http://www.orangepi.org/
- **Ultralytics YOLO**: https://docs.ultralytics.com/

---

## 🚀 下一步行动

### 如果板子还没到手
1. ✅ 代码已完全ready，可以先做其他工作
2. ✅ WSL环境的所有开发工作已完成
3. ✅ 文档和脚本都已准备好
4. ⏸️ 等板子到货后，按照 `docs/guides/BOARD_QUICKSTART.md` 执行即可

### 如果板子已到手
1. 📖 阅读 `docs/guides/BOARD_QUICKSTART.md`（5分钟）
2. 🔌 连接板子网络，获取IP
3. 📦 运行 `scripts/deploy/pack_for_board.sh`打包
4. ⬆️ SCP传输到板子
5. 🔧 板子上运行 `install_dependencies.sh`
6. 🏃 首次推理测试
7. 📊 性能基准测试
8. 🌐 双网卡配置与验证
9. 📸 收集答辩材料

---

**所有资源已准备完毕，祝部署顺利！** 🎉
