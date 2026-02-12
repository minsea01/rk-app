# RK3588硬件集成手册

**目的：** 规范硬件接入时的集成流程，确保无缝从PC验证过渡到硬件部署
**受众：** 硬件开发团队、系统集成工程师
**前置条件：** Phase 1 PC离线验证已完成

---

## 📋 硬件集成前置检查

### 必备条件

```
硬件清单:
☐ RK3588开发板 (Ubuntu 22.04)
☐ 工业摄像头 (支持GigE Vision或CSI接口)
☐ 网络设备 (双Gigabit网卡或RGMII接口)
☐ 电源适配器 (推荐12V 5A)

开发工具:
☐ SSH访问权限
☐ 串口调试器 (可选, 用于启动日志)
☐ 网络分析工具 (iperf3, tcpdump)

环境:
☐ 板载RKNN NPU驱动已安装 (/dev/rknn_0可访问)
☐ 网络连接正常
☐ 内核版本 ≥5.10
```

### 验证环境就绪

```bash
# SSH连接到板子
ssh user@192.168.1.100

# 验证NPU驱动
ls -la /dev/rknn_0
# Expected: crw-rw-rw- ... /dev/rknn_0

# 验证内核版本
uname -r
# Expected: 5.10 or higher

# 验证网络接口
ip link show
# Expected: eth0 (camera), eth1 (detection output)
```

---

## 🔧 第一阶段：基础环境配置 (Day 1-2)

### 1.1 系统初始化

```bash
# 1. 更新系统包
sudo apt-get update
sudo apt-get upgrade -y

# 2. 安装依赖
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    libopencv-dev \
    libssl-dev \
    cmake

# 3. 创建工作目录
mkdir -p /opt/rk-detection
cd /opt/rk-detection

# 4. 克隆项目
git clone <repo-url> .
```

### 1.2 Python环境配置

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 安装基础运行时依赖（轻量）
pip install --upgrade pip
pip install -r requirements.txt

# 3. 安装板端依赖（含 RKNN Lite）
pip install -r requirements_board.txt

# 4. 验证RKNN工具包（Lite）
python3 -c "from rknnlite.api import RKNNLite; print('RKNN Lite OK')"

# 5. 设置PYTHONPATH
export PYTHONPATH=/opt/rk-detection:$PYTHONPATH
```

### 1.3 编译C++二进制 (可选, 推荐)

```bash
cd /opt/rk-detection

# 1. 编译ARM64发布版本
cmake --preset arm64-release -DENABLE_RKNN=ON
cmake --build --preset arm64 -j4

# 2. 验证二进制
file out/arm64/bin/detect_cli
# Expected: ELF 64-bit LSB executable ARM aarch64

# 3. 测试单帧推理
./out/arm64/bin/detect_cli --config config/detection/detect_board.yaml
```

### 1.4 模型部署

```bash
# 1. 验证RKNN模型已复制
ls -lh artifacts/models/*.rknn

# 2. 验证模型大小 (<5MB)
du -h artifacts/models/best.rknn
# Expected: 4.7M

# 3. 验证配置文件
cat config/detection/detect_board.yaml
```

---

## 📸 第二阶段：摄像头集成 (Day 2-3)

### 2.1 硬件摄像头接口实现

当前代码骨架已预留，Phase 2需填充:

```python
# apps/input_source.py - 待硬件到达时实现

class RK3588CameraSource(InputSource):
    """RK3588硬件摄像头驱动"""

    def __init__(self, csi_port=0, resolution=(1920, 1080), fps=30):
        """
        Args:
            csi_port: CSI摄像头端口 (0 或 1)
            resolution: 输出分辨率 (W, H)
            fps: 帧率

        支持的摄像头类型:
        - GigE Vision (工业相机, 通过网口)
        - CSI (显示连接, 板载)
        - USB (USB摄像头)
        """
        self.csi_port = csi_port
        self._resolution = resolution
        self._fps = fps
        self.camera = None

    def open(self):
        """打开摄像头"""
        try:
            # 选项A: 使用rkmedia (RK官方库)
            # from rkmedia import RKCamera
            # self.camera = RKCamera(self.csi_port)

            # 选项B: 使用V4L2 (更兼容)
            import v4l2capture
            device = f'/dev/video{self.csi_port}'
            self.camera = v4l2capture.Video_device(device)
            self.camera.set_format(*self._resolution, fourcc='YUYV')
            self.camera.create_buffers(4)
            self.camera.queue_all_buffers()
            self.camera.start()

        except Exception as e:
            logger.error(f"Failed to open camera: {e}")
            raise

    def read(self) -> tuple:
        """读取一帧"""
        try:
            # V4L2读取
            select.select((self.camera,), (), ())
            image_data = self.camera.read_and_queue()

            # 解码YUYV → BGR
            frame = decode_yuyv(image_data, *self._resolution)

            metadata = {
                'timestamp': time.time(),
                'frame_id': self.frame_count,
                'resolution': self._resolution,
                'source': 'hardware_camera'
            }

            self.frame_count += 1
            return frame, metadata

        except Exception as e:
            logger.error(f"Read error: {e}")
            return None, None

    def close(self):
        """关闭摄像头"""
        if self.camera:
            self.camera.close()

    @property
    def fps(self):
        return self._fps

    @property
    def resolution(self):
        return self._resolution
```

### 2.2 摄像头驱动验证

```bash
# 1. 列出可用摄像头
ls -la /dev/video*

# 2. 测试摄像头 (使用ffmpeg)
ffplay /dev/video0  # 实时预览

# 3. 捕获测试图像
ffmpeg -f v4l2 -i /dev/video0 -vframes 1 test_frame.jpg

# 4. 运行推理测试
python apps/yolov8_rknn_infer.py \
    --config config/detection/detect_board_debug.yaml \
    --max-frames 5
```

### 2.3 GigE Vision工业相机支持 (可选)

```python
# apps/input_source.py - 可选增强

class GigEVisionSource(InputSource):
    """支持GigE Vision工业相机 (网口传输)"""

    def __init__(self, camera_ip: str, port: int = 3956):
        """
        工业相机通过网口传输实时视频

        Args:
            camera_ip: 相机IP地址 (e.g., 192.168.1.101)
            port: GigE Vision端口 (默认3956)
        """
        self.camera_ip = camera_ip
        self.port = port
        self.camera = None

    def open(self):
        """连接GigE Vision相机"""
        try:
            # 选项: 使用PyGEV或OpenCV
            # import pygev
            # cameras = pygev.scan()
            # self.camera = cameras[0]

            # 简化方式: 使用OpenCV + RTSP
            rtsp_url = f"rtsp://{self.camera_ip}/stream"
            self.cap = cv2.VideoCapture(rtsp_url)

        except Exception as e:
            logger.error(f"Failed to connect camera: {e}")
            raise

    def read(self):
        ret, frame = self.cap.read()
        if ret:
            return frame, {'timestamp': time.time()}
        return None, None

    def close(self):
        if self.cap:
            self.cap.release()
```

### 2.4 配置示例

```yaml
# config/detection/detect_board.yaml
# 硬件摄像头配置

source:
  type: hardware_camera    # or gige_vision
  csi_port: 0              # CSI端口0或1
  resolution: [1920, 1080] # 分辨率
  fps: 30                  # 帧率
  # gige_vision配置 (可选)
  camera_ip: 192.168.1.101 # 工业相机IP
  camera_port: 3956        # GigE Vision端口

engine:
  type: rknn
  model: artifacts/models/best.rknn
  imgsz: 416

nms:
  conf_thres: 0.5
  iou_thres: 0.5

output:
  type: tcp
  ip: 192.168.2.1  # 远程监控/存储服务器
  port: 9000
```

---

## 🌐 第三阶段：网络验证 (Day 3-4)

### 3.1 双网口配置

```bash
# 配置脚本已预留, 硬件到达时运行
sudo ./scripts/deploy/configure_dual_nic.sh

# 验证配置
ip addr show
# Expected:
# eth0: 192.168.1.100/24 (camera input)
# eth1: 192.168.2.100/24 (detection output)

# 持久化配置 (可选)
sudo netplan apply
```

### 3.2 网络吞吐测试

```bash
# 1. 测试Port 1 (摄像头输入)
iperf3 -c <camera_server> -B 192.168.1.100 -t 10
# Expected: ≥900 Mbps

# 2. 测试Port 2 (结果输出)
iperf3 -c <result_server> -B 192.168.2.100 -t 10
# Expected: ≥900 Mbps

# 3. 网络捕包分析 (可选)
tcpdump -i eth0 -w eth0_traffic.pcap
tcpdump -i eth1 -w eth1_traffic.pcap
```

### 3.3 TCP结果接收测试

```bash
# 1. 启动结果接收服务器
python scripts/results_receiver.py

# 2. 发送推理结果到TCP
python apps/yolov8_rknn_infer.py \
    --config config/detection/detect_board.yaml \
    --output-host 127.0.0.1 \
    --output-port 9000

# 3. 验证结果被接收
ls -l artifacts/detection_results/
# Expected: 检测结果JSON文件
```

---

## 📊 第四阶段：性能验证 (Week 1)

### 4.1 单帧延迟测试

```bash
# 运行性能测试
python scripts/generate_performance_report.py \
    --model artifacts/models/best.rknn \
    --imgsz 416 \
    --runs 50 \
    --output artifacts/hardware_performance.md

# 预期结果:
# 单帧延迟: 20-30 ms (vs PC 60ms)
# FPS: 33-50 (vs PC 16.4)
# 加速比: 2-3x
```

### 4.2 系统资源监控

```bash
# 1. CPU使用率
top -p $(pgrep -f yolov8_rknn)

# 2. 温度监控
watch -n 1 'cat /sys/class/thermal/thermal_zone*/temp'
# Expected: <60°C

# 3. 功耗监控 (如果板子支持)
cat /sys/class/power_supply/*/power_now
# Expected: <10W

# 4. 内存使用
free -h
```

### 4.3 端到端延迟分析

```yaml
# 添加详细时间戳统计
# config/detection/detect_board_debug.yaml

perf:
  detailed_timing: true
  # 将输出以下指标:
  # - 采集延迟 (摄像头 → 预处理)
  # - 预处理延迟
  # - 推理延迟
  # - 后处理延迟
  # - 网络发送延迟
  # - 总端到端延迟
```

---

## 🎯 第五阶段：精度评估 (Week 2)

### 5.1 准备验证数据集

```bash
# 选项A: 使用COCO val2017 (公开)
cd datasets
# 下载COCO数据集的person类别子集

# 选项B: 使用自建数据集
# 准备带标注的行人检测数据集
# 格式: COCO JSON annotations

# 配置数据集路径
export DATASET_PATH=/path/to/pedestrian_dataset
```

### 5.2 mAP@0.5评估

```bash
# 运行精度评估
python scripts/evaluate_map.py \
    --rknn artifacts/models/best.rknn \
    --dataset $DATASET_PATH \
    --annotations instances.json \
    --output artifacts/hardware_mAP.json

# 预期结果:
# mAP@0.5: >90% (毕业设计要求)
# 如果低于90%, 需要:
# - 模型重训练
# - 微调阈值
# - 数据集增强
```

### 5.3 ONNX vs RKNN对比

```bash
# 生成完整对比报告
python scripts/compare_onnx_rknn.py \
    --onnx artifacts/models/best.onnx \
    --rknn artifacts/models/best.rknn \
    --dataset $DATASET_PATH \
    --output artifacts/onnx_vs_rknn_hardware.json

# 验证精度一致性:
# - 数值差异 <5%
# - 检测框IoU >0.95
# - 类别准确度 >99%
```

---

## 📝 第六阶段：系统验证 (Week 2-3)

### 6.1 稳定性测试

```bash
# 1. 长时间运行测试 (4-8小时)
timeout 28800 python apps/yolov8_rknn_infer.py \
    --config config/detection/detect_board.yaml \
    --log-file artifacts/stability_test.log

# 2. 监控日志
tail -f artifacts/stability_test.log | grep -E "ERROR|WARNING|FPS"

# 3. 统计指标
# - 总帧数
# - 平均FPS
# - 最大延迟
# - 错误次数
# - 内存泄漏 (valgrind, 可选)
```

### 6.2 并发连接测试

```bash
# 模拟多个客户端读取结果
for i in {1..5}; do
    python scripts/results_receiver.py --client-id $i &
done

# 验证:
# - 所有客户端都能接收结果
# - 无丢包现象
# - 延迟不超过阈值
```

### 6.3 容错能力验证

```yaml
# config/detection/detect_board_fault_test.yaml
# 测试故障恢复

network:
  timeout: 5s      # 网络超时处理
  retry: 3         # 重试次数
  fallback: file   # 故障回退到文件保存

inference:
  fallback_engine: onnx  # 如果RKNN失败,使用ONNX

monitoring:
  alert_on_error: true
  restart_on_failure: true
```

---

## 📋 验证清单 (Hardware Integration Checklist)

```
第一阶段: 基础环境 (Day 1-2)
☐ Ubuntu 22.04已安装
☐ NPU驱动 (/dev/rknn_0) 可访问
☐ 网络连接正常
☐ Python虚拟环境就绪
☐ RKNN工具包验证通过
☐ C++二进制编译成功
☐ 模型文件已复制 (<5MB)

第二阶段: 摄像头集成 (Day 2-3)
☐ 摄像头硬件连接
☐ 驱动程序验证 (/dev/video0)
☐ 摄像头图像采集测试
☐ 单帧推理运行成功
☐ 实时推理帧率 >30

第三阶段: 网络验证 (Day 3-4)
☐ 双网口配置完成
☐ eth0: 192.168.1.100/24
☐ eth1: 192.168.2.100/24
☐ 网络吞吐测试 (≥900Mbps)
☐ TCP结果接收测试通过

第四阶段: 性能验证 (Week 1)
☐ 单帧延迟 <50ms
☐ FPS >30
☐ 温度 <60°C
☐ 功耗 <10W
☐ 性能报告已生成

第五阶段: 精度验证 (Week 2)
☐ 测试数据集已准备
☐ mAP@0.5 >90%
☐ ONNX vs RKNN对比 <5%差异
☐ 精度报告已生成

第六阶段: 系统验证 (Week 2-3)
☐ 24小时稳定性测试通过
☐ 多客户端并发连接正常
☐ 错误恢复机制验证
☐ 完整系统日志已保存

最终验收 (Defense准备)
☐ 所有指标达到要求
☐ 性能数据完整
☐ 论文数据已整合
☐ 硬件演示脚本就绪
```

---

## 🚨 常见问题排查

### Q1: NPU推理速度达不到预期

**现象:** 实际30-40ms, 预期20-30ms

**排查:**
```bash
# 1. 检查模型是否为416×416
file artifacts/models/best.rknn
# 如果使用640×640, 会有Transpose CPU fallback

# 2. 检查CPU利用率
top -p $(pgrep detect_cli)
# 如果CPU占用高, 说明有CPU fallback

# 解决方案:
# 使用 yolo11n_416.rknn 替代 best.rknn
# 编辑 config/detection/detect_board.yaml:
# engine.model: artifacts/models/yolo11n_416.rknn
# engine.imgsz: 416
```

### Q2: 精度低于90%

**现象:** mAP@0.5 < 90%

**排查:**
```bash
# 1. 确认数据集格式正确
python -c "import json; json.load(open('annotations.json'))"

# 2. 对比ONNX和RKNN
python scripts/compare_onnx_rknn.py \
    --dataset test_dataset \
    --output comparison.json

# 3. 查看具体误差
# 如果RKNN严重低于ONNX, 可能是量化问题
# 解决方案: 重新训练或增加更多校准图像

# 4. 尝试调整阈值
# config/detection/detect_board.yaml
# nms.conf_thres: 0.4  (降低以增加召回)
```

### Q3: 网络吞吐不足900Mbps

**现象:** iperf3显示 <500Mbps

**排查:**
```bash
# 1. 检查网卡配置
ethtool eth0
# 应该显示: Speed: 1000Mb/s

# 2. 检查驱动版本
ethtool -i eth0

# 3. 网络干扰排查
tcpdump -i eth0 -c 100 | grep -E "error|drop"

# 4. 更新驱动
# 联系Rockchip获取最新网卡驱动

# 临时方案: 使用单网口 (eth0或eth1) 承载所有流量
```

---

## 📚 参考文档

| 文档 | 说明 |
|------|------|
| OFFLINE_PIPELINE_INTEGRATION.md | 流水线架构设计 |
| QUICK_START_PHASE2.md | 快速开始指南 |
| CLAUDE.md | 开发指南 |
| artifacts/board_ready_report.md | 部署就绪报告 |

---

**版本:** 1.0
**最后更新:** 2025-10-30
**状态:** Phase 1完成，待硬件到达
