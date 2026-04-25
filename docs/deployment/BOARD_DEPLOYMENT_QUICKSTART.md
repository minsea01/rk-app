# 🚀 RK3588板上部署快速指南

**目标**: 30分钟内完成基础推理验证
**前提**: 已有RK3588开发板（如Radxa ROCK 5B）

---

## 📋 准备清单

### 硬件
- [ ] RK3588开发板（16GB RAM）
- [ ] 电源适配器（12V/2A或更高）
- [ ] MicroSD卡（64GB+）或eMMC
- [ ] 网线（用于SSH连接）
- [ ] （可选）HDMI显示器

### 软件
- [ ] Ubuntu 20.04/22.04镜像（RK3588适配版）
- [ ] SSH客户端（PC上）
- [ ] 本项目代码包

---

## ⚡ 30分钟快速部署

### Step 1: 连接板子（5分钟）

```bash
# 在PC上，通过SSH连接到板子
# 假设板子IP是192.168.1.100
ssh radxa@192.168.1.100
# 默认密码通常是: radxa 或 rock

# 验证系统
uname -a
# 应该看到类似: Linux rock-5b 5.10.xxx aarch64

# 检查NPU驱动
ls /dev/rknpu*
# 应该看到: /dev/rknpu0 (或类似设备)
```

---

### Step 2: 传输代码（5分钟）

**方案A: 使用rsync（推荐）**
```bash
# 在PC上执行
cd /home/user/rk-app

# 打包必要文件
tar czf rk-app-minimal.tar.gz \
  apps/ \
  tools/convert_onnx_to_rknn.py \
  scripts/deploy/ \
  scripts/profiling/ \
  config/ \
  artifacts/models/best_person_aug_416_norm_int8.rknn \
  --exclude='__pycache__'

# 传输
scp rk-app-minimal.tar.gz radxa@192.168.1.100:/home/radxa/

# 在板子上解压
ssh radxa@192.168.1.100
cd /home/radxa
tar xzf rk-app-minimal.tar.gz
```

**方案B: 使用Git（如有仓库）**
```bash
# 在板子上执行
git clone <your-repo-url> /home/radxa/rk-app
cd rk-app
```

---

### Step 3: 安装依赖（10分钟）

```bash
# 在板子上执行
cd /home/radxa/rk-app

# 更新包管理器
sudo apt update

# 安装Python和基础工具
sudo apt install -y python3 python3-pip

# 检查Python版本
python3 --version
# 如果是3.8+就可以

# 安装核心依赖
pip3 install numpy opencv-python-headless pillow

# 安装RKNN板上运行时（关键）
# 方案1: 从PyPI安装（如果可用）
pip3 install rknn-toolkit-lite2

# 方案2: 从Rockchip官方下载wheel
# 访问: https://github.com/rockchip-linux/rknn-toolkit2/releases
# 下载对应的wheel文件，例如:
# wget https://github.com/.../rknn_toolkit_lite2-1.6.0-cp310-cp310-linux_aarch64.whl
# pip3 install rknn_toolkit_lite2-1.6.0-cp310-cp310-linux_aarch64.whl
```

---

### Step 4: 首次推理测试（5分钟）

```bash
# 在板子上执行
cd /home/radxa/rk-app

# 设置Python路径
export PYTHONPATH=/home/radxa/rk-app

# 下载测试图片（如果没有）
wget -O test.jpg https://ultralytics.com/images/zidane.jpg

# 运行推理
python3 apps/yolov8_rknn_infer.py \
  --model artifacts/models/best_person_aug_416_norm_int8.rknn \
  --source test.jpg \
  --save result.jpg \
  --imgsz 640 \
  --conf 0.25

# 预期输出:
# [INFO] Loading RKNN: artifacts/models/best_person_aug_416_norm_int8.rknn
# [INFO] Initializing runtime, core_mask=0x7
# [INFO] Inference time: XX.XX ms
# [INFO] Detections: X
# [INFO] Saved: result.jpg
```

---

### Step 5: 验证结果（5分钟）

```bash
# 检查输出文件
ls -lh result.jpg
# 应该看到生成的图片文件

# 查看FPS（如果有摄像头）
python3 apps/yolov8_rknn_infer.py \
  --model artifacts/models/best_person_aug_416_norm_int8.rknn \
  # 会自动使用/dev/video0

# 或用测试脚本
python3 scripts/profiling/performance_profiler.py \
  --model artifacts/models/best_person_aug_416_norm_int8.rknn \
  --model-type rknn \
  --images-dir <your_test_images> \
  --limit 100
```

---

## 🔧 常见问题处理

### 问题1: rknn-toolkit-lite2安装失败

**症状**:
```
ERROR: Could not find a version that satisfies the requirement rknn-toolkit-lite2
```

**解决**:
```bash
# 需要手动下载wheel文件
# 访问Rockchip官方仓库
cd /tmp
wget https://github.com/rockchip-linux/rknn-toolkit2/releases/download/v1.6.0/rknn_toolkit_lite2-1.6.0-cp310-cp310-linux_aarch64.whl

# 安装
pip3 install rknn_toolkit_lite2-1.6.0-cp310-cp310-linux_aarch64.whl
```

---

### 问题2: 找不到NPU设备

**症状**:
```
ls /dev/rknpu*
ls: cannot access '/dev/rknpu*': No such file or directory
```

**解决**:
```bash
# 加载NPU驱动
sudo modprobe rknpu

# 如果还是没有,检查内核版本
uname -r
# RK3588需要5.10+内核

# 检查dmesg
dmesg | grep -i rknpu
# 查看是否有错误信息
```

---

### 问题3: OpenCV导入错误

**症状**:
```
ImportError: libGL.so.1: cannot open shared object file
```

**解决**:
```bash
# 安装OpenCV系统依赖
sudo apt install -y libgl1-mesa-glx libglib2.0-0

# 或使用headless版本（已在Step 3中）
pip3 install opencv-python-headless
```

---

### 问题4: 内存不足

**症状**:
```
RuntimeError: Cannot allocate memory
```

**解决**:
```bash
# 检查内存使用
free -h

# 清理内存
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# 减小batch size或图片尺寸
python3 apps/yolov8_rknn_infer.py \
  --model artifacts/models/best_person_aug_416_norm_int8.rknn \
  --source test.jpg \
  --imgsz 416  # 使用更小尺寸
```

---

## 📊 性能验证

### FPS测试

```bash
# 使用性能分析脚本
cd /home/radxa/rk-app

python3 scripts/profiling/performance_profiler.py \
  --model artifacts/models/best_person_aug_416_norm_int8.rknn \
  --model-type rknn \
  --images-dir <test_images_directory> \
  --limit 100 \
  --output artifacts/board_performance.json

# 查看结果
cat artifacts/board_performance.json | python3 -m json.tool
```

**预期性能**:
- 推理延迟: 20-40ms @ 640×640
- FPS: 25-35 (INT8量化)
- 内存峰值: ~300MB

---

### 网络吞吐量测试

```bash
# 在板子上运行
sudo ./scripts/network/rgmii_driver_config.sh

# 配置网口
sudo ip addr add 192.168.1.10/24 dev eth0
sudo ip link set eth0 up

# 900Mbps验证
./scripts/network/network_throughput_validator.sh
```

---

## 🎯 毕设验证清单

完成以下步骤即可满足毕设要求：

### 核心指标
- [ ] 推理成功运行（截图+日志）
- [ ] FPS ≥ 30（性能报告）
- [ ] 延迟 ≤ 45ms（性能报告）
- [ ] 模型体积 ≤ 5MB（ls -lh best_person_aug_416_norm_int8.rknn）
- [ ] 检测类别 > 10（COCO 80类）

### 网络指标
- [ ] 双网口识别（rgmii_driver_config.sh输出）
- [ ] 吞吐量 ≥ 900Mbps（network_throughput_validator.sh报告）

### 行人检测
- [ ] mAP ≥ 90%（需要行人数据集）

---

## 📝 收集证据

### 推理成功证据
```bash
# 截图推理输出
python3 apps/yolov8_rknn_infer.py ... | tee inference.log

# 保存性能数据
python3 scripts/profiling/performance_profiler.py ... \
  --output artifacts/board_performance.json
```

### 网络验证证据
```bash
# RGMII驱动验证
sudo ./scripts/network/rgmii_driver_config.sh > rgmii_report.txt 2>&1

# 吞吐量测试
./scripts/network/network_throughput_validator.sh
# 报告保存在 artifacts/network_reports/
```

### 拍照记录
1. 板子运行时的照片
2. 串口/HDMI输出的照片
3. 检测结果的照片

---

## ⏱️ 时间表

| 步骤 | 预期时间 | 关键任务 |
|------|----------|---------|
| 1. 连接板子 | 5分钟 | SSH连接，环境检查 |
| 2. 传输代码 | 5分钟 | rsync或git clone |
| 3. 安装依赖 | 10分钟 | pip install（可能更久） |
| 4. 首次推理 | 5分钟 | 单张图片测试 |
| 5. 性能验证 | 5分钟 | FPS测试 |
| **总计** | **30分钟** | **基础验证完成** |

**扩展验证**（答辩需要）:
- 网络配置测试: +20分钟
- 行人mAP验证: +1小时（需数据集）
- 撰写实验报告: +2小时

**完整验证总时间**: **4-5小时**

---

## 🆘 紧急联系

**如果遇到问题**:
1. 查看错误日志: `dmesg`, `journalctl -xe`
2. 检查Python导入: `python3 -c "from rknnlite.api import RKNNLite"`
3. 验证NPU: `ls /dev/rknpu*`
4. 查看资源: `top`, `free -h`

**Rockchip官方资源**:
- GitHub: https://github.com/rockchip-linux/rknn-toolkit2
- 文档: https://github.com/rockchip-linux/rknn-toolkit2/tree/master/doc
- 示例: https://github.com/rockchip-linux/rknn-toolkit2/tree/master/rknpu2/examples

---

**准备好了吗？开始部署！** 🚀
