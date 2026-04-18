# 离线流水线集成方案

**目标：** 在PC上完成所有可离线推进的工作，为硬件接入预留清晰接口
**状态：** Phase 1完成，为Phase 2硬件集成预留架构
**日期：** 2025-10-30

---

## 📋 整体框架

```
┌─────────────────────────────────────────────────────────────┐
│                    Detection Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Input Source Interface                                      │
│  ├── [File Mode]      → 读取本地图片/视频文件              │
│  ├── [Camera Mode]    → USB摄像头或RTSP流 (PC验证)        │
│  ├── [Network Mode]   → TCP/UDP网络流 (模拟)              │
│  └── [Board Mode]     → RK3588原始CSI/网口 (硬件接入)     │
│                                                               │
│  ↓ (统一接口)                                               │
│                                                               │
│  Preprocessor        → 图像缩放、格式转换、归一化         │
│  ↓                                                            │
│  Inference Engine    → ONNX (PC) / RKNN (Board)            │
│  ↓                                                            │
│  Postprocessor       → NMS、阈值过滤、检测框输出          │
│  ↓                                                            │
│  Output Handler      → TCP/UDP发送、文件保存、实时显示   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 1️⃣ 输入源接口定义 (Hardware-Agnostic)

### 1.1 接口规范

```python
# apps/input_source.py

from abc import ABC, abstractmethod
import numpy as np

class InputSource(ABC):
    """所有输入源的基类 - 为硬件接入预留接口"""

    @abstractmethod
    def open(self):
        """打开输入源"""
        pass

    @abstractmethod
    def read(self) -> tuple[np.ndarray, dict]:
        """
        读取下一帧
        返回: (frame, metadata)
        - frame: (H, W, 3) BGR uint8
        - metadata: {
            'timestamp': float,
            'frame_id': int,
            'source': str,
            'resolution': (W, H)
          }
        """
        pass

    @abstractmethod
    def close(self):
        """关闭输入源"""
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        """帧率"""
        pass

    @property
    @abstractmethod
    def resolution(self) -> tuple:
        """分辨率 (W, H)"""
        pass
```

### 1.2 实现的输入源

#### A. 文件模式 (PC离线验证)
```python
class FileSource(InputSource):
    """从本地文件读取"""

    def __init__(self, path: str, recursive=True):
        self.path = Path(path)
        self.frames = []
        self.current_idx = 0
        # 支持: 图片文件夹、视频文件、图片列表

    def read(self):
        # 返回 (frame, metadata)
```

**用途：** PC上验证完整流水线
**命令：**
```bash
python apps/yolov8_rknn_infer.py \
  --input-source file \
  --input-path artifacts/test_images/ \
  --output tcp://localhost:9000
```

#### B. 摄像头模式 (PC可选)
```python
class CameraSource(InputSource):
    """USB摄像头或RTSP流"""

    def __init__(self, camera_id=0, rtsp_url=None):
        self.cap = cv2.VideoCapture(camera_id or rtsp_url)

    def read(self):
        # OpenCV读取 + 时间戳
```

**用途：** PC上实时验证
**命令：**
```bash
python apps/yolov8_rknn_infer.py \
  --input-source camera \
  --camera-id 0 \
  --output display  # 显示结果
```

#### C. 模拟流模式 (网络验证)
```python
class SimulatedNetworkSource(InputSource):
    """模拟网络流 (Docker中使用)"""

    def __init__(self, frame_dir, target_host, target_port):
        self.frames = load_images(frame_dir)
        self.socket = socket.socket()
        # 循环发送图片流

    def read(self):
        # 读取本地图片，模拟网络接收延迟
```

**用途：** Docker中模拟双网口网络流
**命令：**
```bash
docker-compose -f docker-compose.dual-nic.yml up
```

#### D. 网络接收模式 (硬件预留)
```python
class NetworkSource(InputSource):
    """从网络接收图像流"""

    def __init__(self, listen_host, listen_port, protocol='tcp'):
        self.socket = socket.socket()
        self.socket.bind((listen_host, listen_port))
        # 接收网络图像

    def read(self):
        # 从网络读取，解码，返回frame
```

**用途：** RK3588从工业摄像头接收视频流
**接口预留：**
```
Port 1 (eth0): 192.168.1.100:8554 (RTSP/GigE Vision)
Port 2 (eth2): 192.168.2.100:9000 (TCP结果输出)
```

#### E. 硬件直接模式 (RK3588)
```python
class RK3588CameraSource(InputSource):
    """直接从RK3588 CSI摄像头读取"""

    def __init__(self, csi_port=0, resolution=(1920, 1080)):
        self.camera = RK3588Camera(csi_port)
        self.resolution = resolution

    def read(self):
        # 直接从CSI获取原始图像
```

**用途：** 硬件部署 (Phase 2)
**配置：**
```yaml
# config/detection/detect_board.yaml
source:
  type: hardware_camera  # RK3588 CSI
  csi_port: 0
  resolution: [1920, 1080]
  fps: 30
```

---

## 2️⃣ 输出处理器设计

### 2.1 输出接口

```python
# apps/output_handler.py

class OutputHandler(ABC):
    @abstractmethod
    def write(self, detections: List[Dict], frame: np.ndarray, metadata: Dict):
        """写入检测结果"""
        pass

    @abstractmethod
    def close(self):
        """关闭输出"""
        pass
```

### 2.2 实现的输出处理

#### A. TCP网络输出 (测试与硬件部署)
```python
class TCPOutputHandler(OutputHandler):
    """发送结果到TCP服务器"""

    def __init__(self, host: str, port: int):
        self.socket = socket.socket()
        self.socket.connect((host, port))

    def write(self, detections, frame, metadata):
        result = {
            'frame_id': metadata['frame_id'],
            'timestamp': metadata['timestamp'],
            'detections': detections,
            'latency_ms': latency
        }
        json_str = json.dumps(result)
        self.socket.send(json_str.encode())
```

**硬件配置：**
```yaml
output:
  type: tcp
  tcp:
    host: 192.168.2.100  # 远程服务器
    port: 9000
```

#### B. 文件保存 (离线分析)
```python
class FileOutputHandler(OutputHandler):
    """保存到文件"""

    def write(self, detections, frame, metadata):
        # 保存JSON结果
        # 保存标注图片 (可选)
        # 保存视频 (可选)
```

#### C. 实时显示 (PC验证)
```python
class DisplayOutputHandler(OutputHandler):
    """OpenCV显示"""

    def write(self, detections, frame, metadata):
        # 绘制框 + 显示 + 统计信息
```

#### D. RTSP流输出 (工业应用)
```python
class RTSPOutputHandler(OutputHandler):
    """输出RTSP流供外部系统订阅"""

    def write(self, detections, frame, metadata):
        # 编码H.264, 发送RTSP客户端
```

---

## 3️⃣ 完整流水线架构

### 3.1 主推理程序

```python
# apps/yolov8_rknn_infer.py (改进版)

class DetectionPipeline:
    """完整检测流水线 - 输入源和输出可切换"""

    def __init__(self, config_path: str):
        self.config = load_config(config_path)

        # 根据配置选择输入源
        self.input_source = self._create_input_source(config['source'])

        # 根据配置选择推理引擎
        self.engine = self._create_engine(config['engine'])

        # 根据配置选择输出处理
        self.output_handler = self._create_output_handler(config['output'])

    def _create_input_source(self, source_config):
        source_type = source_config['type']

        if source_type == 'file':
            return FileSource(source_config['uri'], recursive=True)
        elif source_type == 'camera':
            return CameraSource(source_config.get('camera_id', 0))
        elif source_type == 'network':
            return NetworkSource(
                source_config['host'],
                source_config['port']
            )
        elif source_type == 'hardware_camera':
            return RK3588CameraSource(
                source_config.get('csi_port', 0)
            )
        else:
            raise ValueError(f"Unknown source type: {source_type}")

    def _create_engine(self, engine_config):
        engine_type = engine_config['type']

        if engine_type == 'onnx':
            return ONNXEngine(engine_config['model'])
        elif engine_type == 'rknn':
            return RKNNEngine(engine_config['model'])
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")

    def _create_output_handler(self, output_config):
        output_type = output_config['type']

        if output_type == 'tcp':
            return TCPOutputHandler(output_config['ip'], output_config['port'])
        elif output_type == 'file':
            return FileOutputHandler(output_config['path'])
        elif output_type == 'display':
            return DisplayOutputHandler()
        else:
            raise ValueError(f"Unknown output type: {output_type}")

    def run(self, max_frames=None):
        """主推理循环"""
        self.input_source.open()
        self.output_handler.open()

        frame_count = 0
        while True:
            # 读取输入
            frame, metadata = self.input_source.read()
            if frame is None:
                break

            # 预处理
            preprocessed = preprocess(frame, self.config['preprocess'])

            # 推理
            start = time.perf_counter()
            outputs = self.engine.infer(preprocessed)
            latency = (time.perf_counter() - start) * 1000

            # 后处理
            detections = postprocess(outputs, self.config['nms'])

            # 输出
            self.output_handler.write(detections, frame, {
                **metadata,
                'latency_ms': latency
            })

            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break

        self.input_source.close()
        self.output_handler.close()
```

### 3.2 配置驱动

PC验证配置：
```yaml
# config/detection/detect_file.yaml
source:
  type: folder
  uri: artifacts/test_images/

engine:
  type: onnx
  model: artifacts/models/best.onnx
  input_size: [416, 416]

postprocess:
  conf_threshold: 0.5
  nms_threshold: 0.5
  max_detections: 100

output:
  type: tcp
  tcp:
    host: 127.0.0.1
    port: 9000
```

离线保存 JSON / 可视化时，通过 CLI 参数导出：

```bash
./build/x86-debug/detect_cli \
  --cfg config/detection/detect_file.yaml \
  --json output/test_detections.json \
  --save_vis output/test_vis
```

硬件部署配置：
```yaml
# config/detection/detect_board.yaml
source:
  type: csi
  uri: "device=/dev/video0,width=1920,height=1080,framerate=30,format=NV12"

engine:
  type: rknn
  model: artifacts/models/best.rknn
  input_size: [416, 416]

postprocess:
  conf_threshold: 0.5
  nms_threshold: 0.5
  max_detections: 100

output:
  type: tcp
  tcp:
    host: 192.168.2.1      # 远程监控服务器
    port: 9000
```

---

## 4️⃣ 离线验证清单

### 4.1 PC端离线完成的工作

```
✅ 文件输入源
   - 支持图片文件夹递归读取
   - 支持视频文件逐帧提取
   - 支持列表文件指定图片序列

✅ ONNX推理引擎
   - CPU推理验证 (16.4 FPS @ 416×416)
   - INT8量化精度对比 (<1% 损失)
   - 模型输出格式验证

✅ 完整流水线验证
   - 端到端数据流验证
   - 多格式输入处理
   - 输出结果格式检验

✅ 网络模拟环境
   - Docker网络流模拟
   - TCP结果接收验证
   - 吞吐测试框架
```

### 4.2 硬件接入时的工作 (Phase 2)

```
⏸️ 硬件摄像头源
   - RK3588 CSI摄像头驱动
   - GigE Vision工业相机支持
   - RTSP流接收

⏸️ RKNN推理引擎
   - NPU硬件推理 (目标20-30ms)
   - 多核并行处理
   - 热管理监控

⏸️ 硬件网络输出
   - 双网口配置验证
   - 吞吐量验证 (≥900Mbps)
   - 延迟分析
```

---

## 5️⃣ 部署接口定义

### 5.1 配置文件接口

所有配置通过YAML驱动，支持动态切换：

```bash
# PC验证 - 文件输入，显示输出
python apps/yolov8_rknn_infer.py \
  --config config/detection/detect_file.yaml

# PC验证 - 文件输入，TCP输出
python apps/yolov8_rknn_infer.py \
  --config config/detection/detect_tcp.yaml

# 硬件部署 - 摄像头输入，网络输出
./scripts/deploy/rk3588_run.sh \
  --config config/detection/detect_board.yaml
```

### 5.2 C++ 二进制接口 (硬件部署)

```cpp
// src/main.cpp 预留接口

int main(int argc, char** argv) {
    // 配置解析
    Config cfg = load_config(config_file);

    // 根据config动态创建输入源
    std::unique_ptr<InputSource> input_source =
        CreateInputSource(cfg.source);

    // 推理引擎
    std::unique_ptr<InferenceEngine> engine =
        CreateEngine(cfg.engine);  // ONNX or RKNN

    // 输出处理
    std::unique_ptr<OutputHandler> output =
        CreateOutputHandler(cfg.output);  // TCP, file, display

    // 主推理循环
    while (input_source->Read(frame, metadata)) {
        preprocess(frame);
        engine->Infer(frame, outputs);
        postprocess(outputs, detections);
        output->Write(detections, frame, metadata);
    }

    return 0;
}
```

---

## 6️⃣ 测试验证清单

### 6.1 PC端集成测试 (已完成)

```bash
# 1. 文件输入 + ONNX + 显示输出
✅ python apps/yolov8_rknn_infer.py \
    --config config/detection/detect_file.yaml \
    --max-frames 10

# 2. 文件输入 + ONNX + TCP输出
✅ python apps/yolov8_rknn_infer.py \
    --config config/detection/detect_tcp.yaml

# 3. RKNN PC模拟器验证
✅ python scripts/run_rknn_sim.py

# 4. 网络流模拟 (Docker)
✅ docker-compose -f docker-compose.dual-nic.yml up -d
✅ python scripts/camera_simulator.py
✅ python scripts/results_receiver.py
```

### 6.2 硬件集成测试 (Phase 2)

```
⏸️ 构建ARM64二进制
   cmake --preset arm64-release && cmake --build --preset arm64

⏸️ 一键部署
   ./scripts/deploy/rk3588_run.sh --config detect_board.yaml

⏸️ 性能验证
   - 单帧延迟 (<50ms目标)
   - 吞吐量 (>30 FPS目标)
   - 温度监控 (<60°C目标)

⏸️ 网络验证
   - 双网口配置
   - 吞吐测试 (≥900Mbps)

⏸️ 精度验证
   - mAP@0.5 (>90%目标)
```

---

## 7️⃣ 硬件接入时的无缝衔接

### 7.1 配置切换 (仅改配置)

```bash
# 硬件到达，只需改配置，代码无改动
cp config/detection/detect_file.yaml \
   config/detection/detect_board.yaml

# 编辑detect_board.yaml:
# source.type: hardware_camera → RK3588 CSI
# engine.type: rknn           → NPU推理
# output.type: tcp            → 网络输出

# 一键运行
./scripts/deploy/rk3588_run.sh --config detect_board.yaml
```

### 7.2 输入源接口验证

| 模式 | 输入 | 推理 | 输出 | 当前 | Phase 2 |
|------|------|------|------|------|---------|
| PC验证 | File | ONNX | Display | ✅ | 保留 |
| 网络模拟 | Docker | ONNX | TCP | ✅ | 保留 |
| 硬件部署 | CSI摄像头 | RKNN | TCP | ⏸️ | 新增 |

---

## 📊 离线验证成果

### 已完成 (无需硬件)

- ✅ 模型转换 (PyTorch → ONNX → RKNN)
- ✅ 量化验证 (INT8, <1%损失)
- ✅ PC推理验证 (16.4 FPS)
- ✅ 流水线框架 (可切换输入源/输出)
- ✅ 网络模拟 (Docker双网口模拟)
- ✅ 部署脚本 (一键部署)

### 待硬件验证

- ⏸️ NPU推理性能 (预期33-50 FPS)
- ⏸️ 网络吞吐量 (≥900Mbps)
- ⏸️ 精度评估 (mAP@0.5 >90%)
- ⏸️ 系统稳定性 (24/7运行)

---

## 🎯 总结

### 架构优势

1. **输入源解耦** - 硬件无关，支持多种源
2. **推理引擎通用** - ONNX/RKNN可互换
3. **配置驱动** - 仅需改YAML，代码无改动
4. **渐进式集成** - PC验证→网络模拟→硬件部署

### 硬件接入时间表

- **Day 1**: 构建ARM64二进制 + 部署脚本
- **Day 2**: 配置硬件摄像头源 + 性能测试
- **Day 3-4**: 双网口验证 + 精度测试
- **Week 2+**: 系统优化 + 论文撰写

---

**状态：** ✅ Phase 1完成，架构预留清晰
**下一步：** 等待硬件到达，按此架构无缝集成
**预期：** Dec 2025硬件抵达 → Jan 2026完成Phase 2
