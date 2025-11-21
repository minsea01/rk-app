# 🚀 RK3588实际板子部署完整清单

**适用场景**: 从WSL开发环境到RK3588硬件板子的完整部署流程

---

## 📦 第一步：准备传输包（在WSL上执行）

### 1.1 已有资源清单

✅ **模型文件** (artifacts/models/):
- `best.rknn` (4.7MB) - YOLO11n INT8量化模型
- `yolo11n_416.rknn` (4.3MB) - 416×416优化版本（推荐，避免Transpose CPU回退）
- `yolo11n_int8.rknn` (4.7MB)

✅ **部署脚本**:
- `scripts/deploy/rk3588_run.sh` - 板上一键运行
- `scripts/deploy/deploy_to_board.sh` - SSH远程部署
- `scripts/deploy/install_dependencies.sh` - 依赖安装
- `scripts/deploy/board_health_check.sh` - 健康检查
- `scripts/deploy/configure_dual_nic.sh` - 双网卡配置

✅ **Python应用**:
- `apps/yolov8_rknn_infer.py` - 主推理程序
- `apps/config.py`, `apps/logger.py`, `apps/exceptions.py`
- `apps/utils/preprocessing.py`, `apps/utils/yolo_post.py`

### 1.2 打包必要文件

使用即将创建的打包脚本：
```bash
cd /home/user/rk-app
bash scripts/deploy/pack_for_board.sh
```

这会生成 `rk-app-board-deploy.tar.gz` (约30MB)

---

## 🔌 第二步：硬件连接

### 2.1 硬件清单
- [ ] RK3588开发板（推荐：Radxa ROCK 5B, Orange Pi 5 Plus）
- [ ] 电源适配器（12V/2A 或 USB-C PD 45W）
- [ ] MicroSD卡（64GB+，已烧录Ubuntu 22.04镜像）
- [ ] 网线（用于SSH连接）
- [ ] （可选）工业相机（用于实时推理测试）
- [ ] （可选）HDMI线+显示器（用于调试）

### 2.2 系统镜像烧录

**推荐镜像**:
- Radxa ROCK 5B: https://wiki.radxa.com/Rock5/downloads
- Orange Pi 5 Plus: http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-Pi-5-plus.html

**烧录工具**:
- balenaEtcher (Windows/Linux): https://www.balena.io/etcher/
- Rufus (Windows): https://rufus.ie/

**验证镜像版本**:
```bash
# SSH到板子后执行
uname -a
# 应该显示: Linux xxx 5.10.xxx aarch64

cat /etc/os-release
# 应该显示: Ubuntu 22.04 或 20.04
```

---

## 📡 第三步：网络连接与SSH

### 3.1 连接方式

**方式A: 路由器DHCP（推荐）**
```bash
# 1. 板子网线连到路由器
# 2. 在路由器管理页面查看板子IP（通常是192.168.1.xxx）
# 3. SSH连接
ssh radxa@192.168.1.100  # 或 rock@xxx, orangepi@xxx
# 默认密码: radxa / rock / orangepi
```

**方式B: 直连PC（需要配置静态IP）**
```bash
# 板子端 (通过HDMI连显示器或串口)
sudo ip addr add 192.168.2.100/24 dev eth0
sudo ip link set eth0 up

# PC端 (WSL2需要通过Windows配置)
# Windows网络适配器设置:
#   IP: 192.168.2.1
#   子网掩码: 255.255.255.0

# 然后SSH连接
ssh radxa@192.168.2.100
```

### 3.2 首次登录配置

```bash
# 更新系统（可选，需要时间）
sudo apt update
# sudo apt upgrade -y  # 耗时较长，可跳过

# 创建工作目录
mkdir -p ~/rk-app
cd ~/rk-app
```

---

## ⬆️ 第四步：传输代码与模型

### 4.1 从WSL传输到板子

**方式A: SCP传输（简单直接）**
```bash
# 在WSL上执行
cd /home/user/rk-app

# 使用打包脚本生成的压缩包
scp rk-app-board-deploy.tar.gz radxa@192.168.1.100:/home/radxa/

# 在板子上解压
ssh radxa@192.168.1.100
cd ~
tar xzf rk-app-board-deploy.tar.gz
cd rk-app
```

**方式B: 使用部署脚本（自动化）**
```bash
# 在WSL上执行
bash scripts/deploy/deploy_to_board.sh --host 192.168.1.100 --user radxa
```

**方式C: Git克隆（如果板子能联网）**
```bash
# 在板子上执行
git clone https://github.com/your-username/rk-app.git ~/rk-app
cd ~/rk-app
```

---

## 🔧 第五步：安装依赖（板子上执行）

### 5.1 运行健康检查

```bash
cd ~/rk-app
bash scripts/deploy/board_health_check.sh
```

如果失败，继续下一步安装依赖。

### 5.2 自动安装依赖

```bash
bash scripts/deploy/install_dependencies.sh
```

这个脚本会：
- 检测ARM64架构
- 配置pip清华镜像
- 安装numpy, opencv, pillow等
- 安装rknn-toolkit-lite2
- 验证NPU初始化

### 5.3 手动验证（可选）

```bash
# 验证Python环境
python3 --version  # 应该≥3.8

# 验证NPU驱动
ls /dev/rknpu*     # 应该看到 /dev/rknpu0

# 验证RKNNLite
python3 -c "from rknnlite.api import RKNNLite; print('OK')"

# 验证NPU初始化
python3 << EOF
from rknnlite.api import RKNNLite
rknn = RKNNLite()
ret = rknn.init_runtime()
print(f"NPU init: {'SUCCESS' if ret == 0 else f'FAILED (ret={ret})'}")
rknn.release()
EOF
```

---

## 🏃 第六步：首次推理测试

### 6.1 单张图片测试

```bash
cd ~/rk-app
export PYTHONPATH=$PWD

# 下载测试图片（如果没有）
wget -O assets/test.jpg https://ultralytics.com/images/zidane.jpg

# 运行推理（使用Python）
python3 apps/yolov8_rknn_infer.py \
  --model artifacts/models/yolo11n_416.rknn \
  --source assets/test.jpg \
  --save result.jpg \
  --imgsz 416 \
  --conf 0.5

# 预期输出:
# [INFO] Loading RKNN model: artifacts/models/yolo11n_416.rknn
# [INFO] Initializing RKNNLite runtime
# [INFO] NPU core mask: 0x7 (3 cores)
# [INFO] Inference time: 25.3ms
# [INFO] Detections: 2
# [INFO] Saved to: result.jpg
```

### 6.2 使用一键运行脚本

```bash
# 自动选择CLI或Python运行器
bash scripts/deploy/rk3588_run.sh \
  --model artifacts/models/yolo11n_416.rknn \
  --runner python \
  -- --source assets/test.jpg --save result.jpg
```

### 6.3 查看结果

```bash
# 检查输出文件
ls -lh result.jpg

# 如果有显示器，可以用feh/eog查看
# eog result.jpg

# 或传回PC查看
# 在WSL上执行:
scp radxa@192.168.1.100:~/rk-app/result.jpg /tmp/
```

---

## 📊 第七步：性能测试

### 7.1 FPS基准测试

```bash
cd ~/rk-app

# 使用性能分析脚本（需要先创建）
python3 scripts/profiling/board_benchmark.py \
  --model artifacts/models/yolo11n_416.rknn \
  --iterations 100 \
  --imgsz 416

# 预期输出:
# Mean inference time: 22.5ms
# FPS: 44.4
# NPU utilization: 85%
```

### 7.2 端到端延迟测试

```bash
# 包括预处理+推理+后处理+网络传输
python3 scripts/profiling/end_to_end_latency.py \
  --model artifacts/models/yolo11n_416.rknn \
  --source assets/test.jpg \
  --target-host 192.168.2.100 \
  --target-port 8080

# 预期输出:
# Preprocessing: 3.2ms
# Inference: 22.5ms
# Postprocessing: 5.1ms
# Network TX: 8.5ms
# Total: 39.3ms (< 45ms ✅)
```

---

## 🌐 第八步：双网卡配置（毕设要求）

### 8.1 配置RGMII双网卡

```bash
cd ~/rk-app
sudo bash scripts/deploy/configure_dual_nic.sh

# 这会配置:
# eth0: 192.168.1.100/24 (相机输入)
# eth1: 192.168.2.100/24 (检测结果输出)
```

### 8.2 验证网络吞吐量

```bash
# 在另一台PC上运行iperf3 server
# iperf3 -s -p 5201

# 在板子上测试网口1
iperf3 -c <camera_network_server_ip> -B 192.168.1.100 -t 10 -P 4

# 在板子上测试网口2
iperf3 -c <server_ip> -B 192.168.2.100 -t 10 -P 4

# 预期: ≥900Mbps
```

---

## 🎥 第九步：实时流推理（可选）

### 9.1 USB摄像头测试

```bash
# 检测摄像头
ls /dev/video*

# 实时推理
python3 apps/yolov8_rknn_infer.py \
  --model artifacts/models/yolo11n_416.rknn \
  --source 0 \
  --imgsz 416 \
  --conf 0.5 \
  --show  # 如果有显示器
```

### 9.2 工业相机（网口1）→ 推理 → 上传（网口2）

```bash
# 使用即将创建的双网口流水线脚本
bash scripts/deploy/dual_nic_pipeline.sh \
  --input-interface eth0 \
  --input-port 8554 \
  --output-interface eth1 \
  --output-host 192.168.2.200 \
  --output-port 8080 \
  --model artifacts/models/yolo11n_416.rknn
```

---

## ✅ 第十步：毕设验证清单

### 10.1 功能验证

- [ ] **模型加载成功**: RKNNLite初始化返回0
- [ ] **推理正常运行**: 能够处理单张图片
- [ ] **检测结果正确**: 输出bbox坐标、置信度、类别
- [ ] **可视化输出**: 生成标注后的图片

### 10.2 性能指标

- [ ] **模型体积**: ≤5MB (yolo11n_416.rknn = 4.3MB ✅)
- [ ] **FPS**: ≥30 (目标: 35-45 FPS @ 416×416)
- [ ] **延迟**: ≤45ms (端到端)
- [ ] **内存占用**: ≤500MB

### 10.3 网络指标

- [ ] **双网口识别**: eth0 + eth1 都能up
- [ ] **吞吐量**: ≥900Mbps (每个网口)
- [ ] **端口1接收**: 能从相机网络接收1080P流
- [ ] **端口2上传**: 能上传检测结果到服务器

### 10.4 行人检测（核心指标）

- [ ] **mAP@0.5**: ≥90% (需要CityPersons微调)
  - 当前基线: 61.57% (YOLO11n预训练)
  - 路径: 微调CityPersons数据集2-4小时

---

## 📸 第十一步：收集答辩材料

### 11.1 运行截图

```bash
# 推理日志
python3 apps/yolov8_rknn_infer.py ... 2>&1 | tee inference.log

# 性能报告
python3 scripts/profiling/board_benchmark.py ... --output performance.json

# 网络配置
sudo bash scripts/deploy/configure_dual_nic.sh 2>&1 | tee network_config.log

# 吞吐量测试
iperf3 -c <server> ... 2>&1 | tee iperf_eth0.log
```

### 11.2 拍照记录

1. **硬件照片**:
   - 板子整体照片（标注RK3588芯片位置）
   - 双网口连接照片
   - 运行时LED指示灯

2. **软件截图**:
   - SSH终端运行日志
   - 检测结果可视化
   - 性能监控界面（htop, nvidia-smi-like）

3. **实验数据**:
   - FPS曲线图
   - 网络吞吐量表格
   - mAP评估报告

---

## 🔥 常见问题排查

### 问题1: NPU初始化失败 (ret=-1)

```bash
# 检查驱动
lsmod | grep rknpu

# 加载驱动
sudo modprobe rknpu

# 检查设备权限
ls -l /dev/rknpu*
sudo chmod 666 /dev/rknpu0  # 如果权限不够
```

### 问题2: rknn-toolkit-lite2安装失败

```bash
# 手动下载wheel
cd /tmp
PYTHON_VER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
RKNN_VER="1.6.0"
WHEEL="rknn_toolkit_lite2-${RKNN_VER}-${PYTHON_VER}-${PYTHON_VER}-linux_aarch64.whl"

wget https://github.com/rockchip-linux/rknn-toolkit2/releases/download/v${RKNN_VER}/${WHEEL}
pip3 install ${WHEEL}
```

### 问题3: 网口不识别

```bash
# 检查网卡
ip link show

# 检查dmesg
dmesg | grep -i "eth\|rgmii"

# 检查设备树
cat /proc/device-tree/model
# 应该显示RK3588相关型号
```

### 问题4: 推理速度慢 (>100ms)

```bash
# 检查是否用了640×640（会导致Transpose CPU回退）
# 改用416×416
python3 apps/yolov8_rknn_infer.py \
  --model artifacts/models/yolo11n_416.rknn \
  --imgsz 416

# 检查NPU核心数
# 在apps/yolov8_rknn_infer.py中确认:
# rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)  # 使用3个核
```

---

## 📚 参考资料

**Rockchip官方**:
- RKNN-Toolkit2: https://github.com/rockchip-linux/rknn-toolkit2
- 文档: https://github.com/rockchip-linux/rknn-toolkit2/tree/master/doc
- 示例: https://github.com/rockchip-linux/rknn-toolkit2/tree/master/rknpu2/examples

**板子厂商**:
- Radxa ROCK 5B Wiki: https://wiki.radxa.com/Rock5
- Orange Pi 5 Plus: http://www.orangepi.org/

**YOLO相关**:
- Ultralytics Docs: https://docs.ultralytics.com/
- CityPersons Dataset: https://www.cityscapes-dataset.com/

---

## ⏱️ 时间规划

| 阶段 | 时间 | 任务 |
|------|------|------|
| 硬件准备 | 1天 | 购买板子、烧录镜像 |
| 基础部署 | 0.5天 | SSH连接、传输代码、安装依赖 |
| 功能验证 | 0.5天 | 推理测试、性能测试 |
| 网络配置 | 1天 | 双网卡配置、吞吐量测试 |
| 流水线集成 | 1天 | 相机接入、结果上传 |
| mAP微调 | 2-4小时 | CityPersons微调（可选，在PC上） |
| 答辩材料 | 1天 | 截图、拍照、报告 |
| **总计** | **4-5天** | **完整验证** |

---

**准备就绪？开始部署！** 🚀
