# 🚀 RK3588板子部署快速入门

**从WSL到板子的完整部署流程（4个步骤，30分钟内完成首次推理）**

---

## 前置条件

- ✅ RK3588开发板已烧录Ubuntu 22.04镜像
- ✅ 板子已连接网络（能SSH访问）
- ✅ WSL项目代码完整（已有RKNN模型）

---

## 第一步：在WSL上打包（5分钟）

```bash
cd /home/user/rk-app

# 1. 打包必要文件（~30MB压缩包）
bash scripts/deploy/pack_for_board.sh

# 输出: rk-app-board-deploy.tar.gz

# 2. 传输到板子
scp rk-app-board-deploy.tar.gz radxa@<板子IP>:/home/radxa/

# 例如:
# scp rk-app-board-deploy.tar.gz radxa@192.168.1.100:/home/radxa/
```

**包含内容：**
- Python应用代码 (apps/)
- RKNN模型文件 (artifacts/models/*.rknn)
- 部署脚本 (scripts/deploy/)
- 配置文件 (config/)
- 测试图片 (assets/)

---

## 第二步：板子上解压并检查（5分钟）

```bash
# SSH登录板子
ssh radxa@192.168.1.100

# 解压
cd ~
tar xzf rk-app-board-deploy.tar.gz
cd rk-app

# 健康检查（可选但推荐）
bash scripts/deploy/board_health_check.sh

# 如果全部PASS，说明环境ready
# 如果有FAIL，继续第三步安装依赖
```

**健康检查项目：**
- Python3、pip3安装
- NumPy、OpenCV、Pillow
- RKNNLite库
- NPU设备文件 (/dev/rknpu0)
- NPU驱动模块

---

## 第三步：安装依赖（10分钟）

```bash
# 在板子上执行
cd ~/rk-app

# 自动安装所有依赖
bash scripts/deploy/install_dependencies.sh

# 这个脚本会：
# 1. 检测ARM64架构
# 2. 配置pip清华镜像（加速下载）
# 3. 安装numpy, opencv-python-headless, pillow
# 4. 安装rknn-toolkit-lite2
# 5. 验证NPU初始化

# 如果rknn-toolkit-lite2安装失败，需要手动下载wheel：
# wget https://github.com/rockchip-linux/rknn-toolkit2/releases/download/v1.6.0/rknn_toolkit_lite2-1.6.0-cp310-cp310-linux_aarch64.whl
# pip3 install rknn_toolkit_lite2-1.6.0-cp310-cp310-linux_aarch64.whl
```

---

## 第四步：首次推理（5分钟）

```bash
cd ~/rk-app
export PYTHONPATH=$PWD

# 方式A: 使用Python直接运行
python3 apps/yolov8_rknn_infer.py \
  --model artifacts/models/yolo11n_416.rknn \
  --source assets/test.jpg \
  --save result.jpg \
  --imgsz 416 \
  --conf 0.5

# 方式B: 使用一键运行脚本
bash scripts/deploy/rk3588_run.sh \
  --model artifacts/models/yolo11n_416.rknn \
  --runner python \
  -- --source assets/test.jpg --save result.jpg

# 预期输出:
# [INFO] Loading RKNN model: artifacts/models/yolo11n_416.rknn
# [INFO] Initializing RKNNLite runtime
# [INFO] NPU core mask: 0x7 (使用3个NPU核心)
# [INFO] Inference time: 22.5ms
# [INFO] Detections: 2
# [INFO] Saved to: result.jpg
```

**检查结果：**
```bash
# 查看输出图片大小
ls -lh result.jpg

# 传回PC查看（在WSL上执行）
scp radxa@192.168.1.100:~/rk-app/result.jpg /tmp/
```

---

## 进阶：性能测试与验证

### 1. FPS基准测试

```bash
python3 scripts/profiling/board_benchmark.py \
  --model artifacts/models/yolo11n_416.rknn \
  --iterations 100 \
  --imgsz 416 \
  --output artifacts/board_performance.json

# 预期输出:
# Mean latency: 22.5ms
# Mean FPS: 44.4
# ✅ FPS ≥ 30: PASS
# ✅ Latency ≤ 45ms: PASS
```

### 2. 端到端延迟测试

```bash
python3 scripts/profiling/end_to_end_latency.py \
  --model artifacts/models/yolo11n_416.rknn \
  --source assets/test.jpg \
  --imgsz 416 \
  --iterations 100 \
  --output artifacts/e2e_latency.json

# 输出包括:
# - Preprocessing: 3.2ms
# - Inference: 22.5ms
# - Postprocessing: 5.1ms
# - Total: 30.8ms (< 45ms ✅)
```

### 3. 双网卡配置（毕设要求）

```bash
# 配置RGMII双千兆网卡
sudo bash scripts/deploy/configure_dual_nic.sh

# 配置后:
# eth0: 192.168.1.100/24 (相机输入)
# eth1: 192.168.2.100/24 (检测结果输出)

# 验证网络吞吐量（需要另一台PC运行iperf3 server）
# iperf3 -c <server_ip> -B 192.168.1.100 -t 10 -P 4
# iperf3 -c <server_ip> -B 192.168.2.100 -t 10 -P 4
# 预期: ≥900Mbps
```

### 4. 双网口流水线测试

```bash
# 相机流 (eth0) → 推理 → 结果上传 (eth1)
bash scripts/deploy/dual_nic_pipeline.sh \
  --input-source rtsp://192.168.1.100:8554/stream \
  --output-host 192.168.2.200 \
  --output-port 8080 \
  --model artifacts/models/yolo11n_416.rknn

# 或使用USB摄像头测试
bash scripts/deploy/dual_nic_pipeline.sh \
  --input-source 0 \
  --output-host 192.168.2.200
```

---

## 毕设验证清单

### 功能指标
- [ ] 模型成功加载 (RKNNLite初始化成功)
- [ ] 推理正常运行 (单张图片测试通过)
- [ ] 检测结果正确 (输出bbox、置信度、类别)

### 性能指标
- [ ] **模型体积 ≤ 5MB**: yolo11n_416.rknn = 4.3MB ✅
- [ ] **FPS ≥ 30**: 目标35-45 FPS @ 416×416
- [ ] **延迟 ≤ 45ms**: 端到端延迟测试
- [ ] **内存占用 ≤ 500MB**: 监控峰值内存

### 网络指标
- [ ] **双网口识别**: eth0 + eth1都能up
- [ ] **吞吐量 ≥ 900Mbps**: iperf3测试每个网口
- [ ] **端口1接收**: 能从相机网络接收1080P流
- [ ] **端口2上传**: 能上传检测结果到服务器

### 行人检测（核心）
- [ ] **mAP@0.5 ≥ 90%**: 需要CityPersons微调
  - 当前基线: 61.57% (YOLO11n预训练)
  - 微调路径: `bash scripts/train/train_citypersons.sh` (2-4小时)

---

## 常见问题

### NPU初始化失败 (ret=-1)
```bash
# 加载NPU驱动
sudo modprobe rknpu
ls /dev/rknpu*  # 应该看到 /dev/rknpu0

# 检查权限
sudo chmod 666 /dev/rknpu0
```

### OpenCV导入错误
```bash
# 安装依赖
sudo apt install -y libgl1-mesa-glx libglib2.0-0

# 或使用headless版本
pip3 install opencv-python-headless
```

### 推理速度慢 (>100ms)
```bash
# 使用416×416模型（避免Transpose CPU回退）
python3 apps/yolov8_rknn_infer.py \
  --model artifacts/models/yolo11n_416.rknn \
  --imgsz 416
```

### 网口不识别
```bash
# 检查网卡
ip link show

# 检查dmesg
dmesg | grep -i "eth\|rgmii"

# 检查设备树
cat /proc/device-tree/model  # 应显示RK3588型号
```

---

## 快速命令备忘录

```bash
# 【WSL端】打包传输
cd /home/user/rk-app
bash scripts/deploy/pack_for_board.sh
scp rk-app-board-deploy.tar.gz radxa@<IP>:/home/radxa/

# 【板子端】部署
ssh radxa@<IP>
tar xzf rk-app-board-deploy.tar.gz && cd rk-app
bash scripts/deploy/install_dependencies.sh
bash scripts/deploy/board_health_check.sh

# 【板子端】推理
export PYTHONPATH=$PWD
python3 apps/yolov8_rknn_infer.py \
  --model artifacts/models/yolo11n_416.rknn \
  --source assets/test.jpg --save result.jpg

# 【板子端】性能测试
python3 scripts/profiling/board_benchmark.py \
  --model artifacts/models/yolo11n_416.rknn --iterations 100

# 【板子端】双网卡配置
sudo bash scripts/deploy/configure_dual_nic.sh
```

---

## 参考文档

- 完整部署清单: `docs/deployment/RK3588_DEPLOYMENT_CHECKLIST.md`
- 快速部署指南: `docs/deployment/BOARD_DEPLOYMENT_QUICKSTART.md`
- RGMII网络指南: `docs/docs/RGMII_NETWORK_GUIDE.md`
- Rockchip官方: https://github.com/rockchip-linux/rknn-toolkit2

---

**准备好了吗？开始部署！** 🚀

**预计总时间：30分钟（首次推理） + 2小时（完整验证）**
