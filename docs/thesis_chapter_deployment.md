# 第四章 部署实现

## 4.1 部署方案选择

### 4.1.1 方案对比

本项目评估了三种部署方案：

| 方案 | 实现语言 | 部署复杂度 | 性能 | 可维护性 | 最终选择 |
|------|--------|---------|------|---------|----------|
| **A: Python + RKNNLite** | Python | ⭐ 低 | 95% | ⭐⭐⭐ 高 | ✅ 优先 |
| B: C++ + RKNN SDK | C++ | ⭐⭐⭐ 高 | 100% | ⭐ 低 | ⏸️ 可选 |
| C: Docker容器 | Python/C++ | ⭐⭐ 中 | 90% | ⭐⭐ 中 | ⏸️ 备选 |

### 4.1.2 选择Python方案的原因

**技术原因**：
1. RKNNLite (Python) 使用**相同的NPU后端**
   - 推理延迟基本相同 (差异<10%)
   - 最终FPS性能相当

2. **避免交叉编译复杂性**
   - ARM64依赖管理困难 (OpenCV, yaml-cpp)
   - WSL2环境不支持多架构apt包
   - 可以在板上原生编译（如需要）

3. **快速迭代**
   - 修改模型参数无需重新编译
   - 支持动态配置加载
   - 便于调试和优化

**工程原因**：
1. **缩短上市时间**：立即部署，无需等待编译
2. **风险最小**：经过PC模拟验证
3. **易于扩展**：支持多路推理、实时参数调整
4. **便于文档**：代码易读，便于论文展示

**项目约束**：
- C++优化可作为**Phase 3的可选工作**
- 不影响毕业要求
- 可在论文中作为"性能优化方向"讨论

---

## 4.2 Python运行环境准备

### 4.2.1 PC开发环境

**开发机配置**：
```
操作系统：WSL2 Ubuntu 22.04
Python：3.10.12
CUDA：11.7
cuDNN：8.x
```

**虚拟环境设置**：
```bash
# 创建虚拟环境（首次）
python3.10 -m venv ~/yolo_env

# 激活
source ~/yolo_env/bin/activate

# 安装核心依赖
pip install --upgrade pip setuptools wheel

# 安装YOLO和推理框架
pip install ultralytics==8.3.205
pip install rknn-toolkit2==2.3.2
pip install onnxruntime-gpu==1.16.3  # CUDA 11.x兼容版本（不是1.23.x!)

# 安装辅助库
pip install opencv-python==4.11.0.86
pip install numpy==1.26.4
pip install pyyaml
pip install tqdm

# 验证GPU支持
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# 预期输出：['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

### 4.2.2 RK3588板上环境

**系统要求**：
```
操作系统：Ubuntu 20.04 / 22.04 LTS
内核版本：4.19 或更高
内存：≥2GB (推荐4GB以上)
存储：≥1GB (用于模型和临时数据)
```

**首次部署步骤** (当板子到货时)：

```bash
# 1. 登录到RK3588
ssh root@<board_ip>
# 默认密码通常是 root 或 rockchip

# 2. 更新系统包
sudo apt-get update
sudo apt-get upgrade -y

# 3. 安装Python和pip
sudo apt-get install -y python3 python3-pip python3-dev

# 4. 升级pip
pip3 install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 安装RKNNLite (在线安装)
pip3 install rknn-toolkit-lite2 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 6. 安装OpenCV
pip3 install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple

# 7. 验证安装
python3 -c "from rknn.api import RKNN; print('✓ RKNNLite安装成功')"
python3 -c "import cv2; print(f'✓ OpenCV版本: {cv2.__version__}')"
```

---

## 4.3 Python推理框架集成

### 4.3.1 完整推理流程

**文件**：`apps/yolov8_rknn_infer.py`

```python
#!/usr/bin/env python3
"""
RK3588板上YOLO推理框架
支持RKNN模型推理和实时检测结果输出
"""

import cv2
import numpy as np
from rknn.api import RKNN
from apps.config import ModelConfig
from apps.exceptions import InferenceError, PreprocessError
from apps.logger import setup_logger
from apps.utils.yolo_post import postprocess_yolov8

logger = setup_logger(__name__)

class RKNNInferenceEngine:
    """RK3588 RKNN推理引擎"""

    def __init__(self, model_path, target_size=640):
        """
        初始化推理引擎

        参数：
            model_path: RKNN模型文件路径
            target_size: 目标分辨率 (416 或 640)
        """
        self.model_path = model_path
        self.target_size = target_size
        self.rknn = None

        # 初始化模型
        self._init_model()

    def _init_model(self):
        """加载和初始化RKNN模型"""
        try:
            self.rknn = RKNN(verbose=False)

            # 加载预编译的RKNN模型
            ret = self.rknn.load_rknn(self.model_path)
            if ret != 0:
                raise InferenceError(f"加载模型失败: {ret}")

            # 初始化运行时 (自动选择NPU核心)
            ret = self.rknn.init_runtime(core_mask=RKNN_NPU_CORE_AUTO)
            if ret != 0:
                raise InferenceError(f"初始化运行时失败: {ret}")

            logger.info(f"✓ 已加载模型: {self.model_path}")

        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise

    def preprocess(self, image):
        """
        预处理：准备模型输入

        输入：BGR格式图像
        输出：uint8张量 (1, H, W, 3)
        """
        try:
            h, w = image.shape[:2]

            # 1. 缩放（保持宽高比）
            scale = min(self.target_size / h, self.target_size / w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # 2. Letterbox填充
            canvas = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
            top = (self.target_size - new_h) // 2
            left = (self.target_size - new_w) // 2
            canvas[top:top+new_h, left:left+new_w] = resized

            # 3. BGR → RGB
            canvas = canvas[..., ::-1]

            # 4. 返回uint8格式 (NPU输入)
            return canvas.astype(np.uint8), (top, left, scale)

        except Exception as e:
            raise PreprocessError(f"预处理失败: {e}")

    def infer(self, image):
        """
        执行推理

        输入：BGR格式图像
        输出：模型原始输出
        """
        try:
            # 预处理
            input_tensor, meta = self.preprocess(image)

            # 推理 (输入必须是uint8)
            outputs = self.rknn.inference([input_tensor])

            return outputs, meta

        except Exception as e:
            raise InferenceError(f"推理执行失败: {e}")

    def detect(self, image, conf_threshold=0.5, iou_threshold=0.5):
        """
        完整检测流程：预处理 → 推理 → 后处理

        参数：
            image: BGR格式输入图像
            conf_threshold: 置信度阈值 (推荐0.5)
            iou_threshold: NMS IoU阈值

        返回：
            detections: [{'box': [x,y,w,h], 'conf': 0.95, 'cls': 0}, ...]
        """
        try:
            # 1. 推理
            outputs, meta = self.infer(image)

            # 2. 后处理
            detections = postprocess_yolov8(
                outputs[0],
                image.shape[:2],
                meta,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )

            return detections

        except Exception as e:
            logger.error(f"检测失败: {e}")
            return []

    def __del__(self):
        """清理资源"""
        if self.rknn:
            self.rknn.release()


# 快速接口
def run_inference(model_path, image_path, conf_threshold=0.5):
    """
    简单推理接口

    用法：
        detections = run_inference('model.rknn', 'image.jpg', conf_threshold=0.5)
    """
    engine = RKNNInferenceEngine(model_path)

    image = cv2.imread(image_path)
    detections = engine.detect(image, conf_threshold=conf_threshold)

    return detections


if __name__ == '__main__':
    # 示例用法
    engine = RKNNInferenceEngine('artifacts/models/best.rknn', target_size=416)

    # 加载测试图像
    image = cv2.imread('assets/test.jpg')

    # 执行检测
    detections = engine.detect(image, conf_threshold=0.5)

    # 打印结果
    for det in detections:
        print(f"类别: {det['cls']}, 置信度: {det['conf']:.2f}, 框: {det['box']}")
```

### 4.3.2 关键代码细节

#### 后处理模块

```python
# apps/utils/yolo_post.py

def postprocess_yolov8(output, original_shape, meta, conf_threshold=0.5, iou_threshold=0.5):
    """
    YOLO输出解码和后处理

    输入：
        output: 模型输出张量 (1, 84, 3549) 或 (1, 84, 8400)
        original_shape: 原始图像尺寸 (h, w)
        meta: 预处理元数据 (top, left, scale)
        conf_threshold: 置信度过滤阈值
        iou_threshold: NMS阈值

    输出：
        detections: 列表，每个元素为 {'box': [x,y,w,h], 'conf': float, 'cls': int}
    """
    # 1. 解码YOLO头
    boxes, confidences, class_probs = decode_yolo_output(output)

    # 2. 置信度过滤 (关键优化点!)
    # conf < 0.5时，后续NMS需处理大量框，导致性能下降
    # conf >= 0.5时，框数大幅减少，NMS快速
    mask = confidences > conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_probs = class_probs[mask]

    # 3. NMS去重
    keep_indices = nms(boxes, confidences, iou_threshold)
    final_boxes = boxes[keep_indices]
    final_confs = confidences[keep_indices]
    final_classes = np.argmax(class_probs[keep_indices], axis=1)

    # 4. 坐标映射 (从640→原始尺寸)
    top, left, scale = meta
    final_boxes = map_boxes_to_original(final_boxes, top, left, scale)

    # 5. 封装结果
    detections = []
    for box, conf, cls in zip(final_boxes, final_confs, final_classes):
        detections.append({
            'box': box,  # [x, y, w, h]
            'conf': float(conf),
            'cls': int(cls)
        })

    return detections


def nms(boxes, scores, threshold):
    """非极大值抑制 (NMS)"""
    if len(boxes) == 0:
        return []

    # 按置信度降序排列
    indices = np.argsort(-scores)

    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        # 计算当前框与其他框的IoU
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        ious = calculate_iou_batch(current_box, other_boxes)

        # 保留IoU < threshold的框
        mask = ious < threshold
        indices = indices[1:][mask]

    return keep
```

---

## 4.4 部署脚本

### 4.4.1 一键运行脚本

**文件**：`scripts/deploy/rk3588_run.sh`

```bash
#!/bin/bash
set -e

# RK3588一键运行脚本
# 自动检测二进制和Python运行时

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 配置
MODEL_PATH="${PROJECT_ROOT}/artifacts/models/best.rknn"
CONFIG_PATH="${PROJECT_ROOT}/config/detection/detect_rknn.yaml"
RUNNER="${1:-auto}"  # auto|binary|python
MODEL_OVERRIDE="${2:-}"

# 日志
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

# 检查文件
if [ ! -f "$MODEL_PATH" ]; then
    log "错误: 模型不存在: $MODEL_PATH"
    exit 1
fi

# 方案A: 尝试运行C++二进制
if [ "$RUNNER" = "auto" ] || [ "$RUNNER" = "binary" ]; then
    BINARY="${PROJECT_ROOT}/out/arm64/bin/detect_cli"

    if [ -f "$BINARY" ]; then
        log "✓ 检测到ARM64二进制: $BINARY"
        log "启动推理..."

        # 配置运行时环境
        export LD_LIBRARY_PATH="${PROJECT_ROOT}/out/arm64/lib:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH"

        # 运行二进制
        if [ -n "$MODEL_OVERRIDE" ]; then
            "$BINARY" --cfg "$CONFIG_PATH" --model "$MODEL_OVERRIDE"
        else
            "$BINARY" --cfg "$CONFIG_PATH"
        fi
        exit $?
    fi
fi

# 方案B: 回退到Python运行时
log "检测到Python运行时，启动Python推理..."

# 验证Python环境
if ! python3 -c "import rknn" 2>/dev/null; then
    log "错误: RKNNLite未安装，请执行: pip3 install rknn-toolkit-lite2"
    exit 1
fi

# 启动Python推理
export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"

python3 -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')

from apps.yolov8_rknn_infer import RKNNInferenceEngine
from apps.config import get_detection_config
import cv2

# 加载配置
config = get_detection_config(size=416)

# 初始化引擎
log('初始化推理引擎...')
engine = RKNNInferenceEngine(
    model_path='${MODEL_OVERRIDE:-$MODEL_PATH}',
    target_size=416
)

# 循环推理 (示例：从摄像头或视频文件)
# TODO: 集成实际的输入源 (GigE相机 或 RTSP流)

log('✓ 推理引擎就绪，等待输入...')
print('Python RKNNLite推理框架已启动')
" || exit $?
```

### 4.4.2 SSH部署脚本

**文件**：`scripts/deploy/deploy_to_board.sh`

```bash
#!/bin/bash
# SSH远程部署到RK3588

set -e

# 参数
BOARD_IP="${1:-}"
BOARD_USER="${2:-root}"
BOARD_PATH="/opt/rk-app"

if [ -z "$BOARD_IP" ]; then
    echo "用法: $0 <board_ip> [board_user]"
    exit 1
fi

log() {
    echo "[部署] $*"
}

# 检查SSH连接
log "检查SSH连接到 $BOARD_USER@$BOARD_IP ..."
if ! ssh -o ConnectTimeout=5 "$BOARD_USER@$BOARD_IP" "echo ✓ SSH连接成功" 2>/dev/null; then
    log "错误: 无法连接到板子"
    exit 1
fi

# 上传文件
log "上传模型文件..."
scp -r artifacts/models/*.rknn "$BOARD_USER@$BOARD_IP:$BOARD_PATH/models/"

log "上传应用代码..."
scp -r apps/ "$BOARD_USER@$BOARD_IP:$BOARD_PATH/"

log "上传配置文件..."
scp -r config/ "$BOARD_USER@$BOARD_IP:$BOARD_PATH/"

log "上传部署脚本..."
scp scripts/deploy/rk3588_run.sh "$BOARD_USER@$BOARD_IP:$BOARD_PATH/"

# 在板上运行
log "启动推理..."
ssh "$BOARD_USER@$BOARD_IP" "cd $BOARD_PATH && bash ./rk3588_run.sh --runner python"
```

---

## 4.5 网络集成

### 4.5.1 检测结果序列化

**JSON格式输出**：

```python
import json
from datetime import datetime

def serialize_detections_json(detections, frame_id, inference_time_ms):
    """将检测结果序列化为JSON"""

    result = {
        "timestamp": datetime.now().isoformat(),
        "frame_id": frame_id,
        "inference_time_ms": inference_time_ms,
        "detections": []
    }

    for det in detections:
        x, y, w, h = det['box']
        result["detections"].append({
            "class_id": det['cls'],
            "class_name": COCO_CLASSES[det['cls']],
            "confidence": det['conf'],
            "bbox": {
                "x": float(x),
                "y": float(y),
                "width": float(w),
                "height": float(h)
            }
        })

    return json.dumps(result)
```

### 4.5.2 网络发送

```python
import socket
import json

class ResultSender:
    """将检测结果发送到服务器"""

    def __init__(self, host='127.0.0.1', port=9000):
        self.host = host
        self.port = port
        self.socket = None

    def connect(self):
        """建立TCP连接"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))

    def send(self, result_json):
        """发送JSON结果"""
        if not self.socket:
            self.connect()

        message = result_json.encode('utf-8') + b'\n'
        self.socket.sendall(message)

    def close(self):
        """关闭连接"""
        if self.socket:
            self.socket.close()


# 使用示例
sender = ResultSender(host='192.168.1.100', port=9000)
sender.connect()

for frame_id, detections in enumerate(detection_stream):
    result_json = serialize_detections_json(detections, frame_id, 32)
    sender.send(result_json)

sender.close()
```

---

## 4.6 故障排查

### 4.6.1 常见问题

**问题1：模型加载失败**
```
错误: Could not find model file
原因: 模型路径不正确或文件损坏
解决:
  1. 检查路径: file artifacts/models/best.rknn
  2. 验证大小: du -h artifacts/models/best.rknn (应为4.7MB)
  3. 重新转换: python tools/convert_onnx_to_rknn.py
```

**问题2：RKNNLite导入失败**
```
错误: ModuleNotFoundError: No module named 'rknn'
原因: 未安装RKNNLite
解决: pip3 install rknn-toolkit-lite2
```

**问题3：推理特别慢（<1 FPS）**
```
症状: conf阈值设置为0.25
根本原因: NMS处理8400+个框
解决: 更改conf = 0.5 (提升600倍!)
```

### 4.6.2 性能调试

```python
# 性能分析脚本
import time

def benchmark_inference(engine, image, num_runs=10):
    """基准测试推理性能"""

    times = {
        'preprocess': [],
        'inference': [],
        'postprocess': []
    }

    for _ in range(num_runs):
        # 预处理
        start = time.time()
        input_tensor, meta = engine.preprocess(image)
        times['preprocess'].append(time.time() - start)

        # 推理
        start = time.time()
        outputs = engine.rknn.inference([input_tensor])
        times['inference'].append(time.time() - start)

        # 后处理
        start = time.time()
        detections = postprocess_yolov8(outputs[0], image.shape[:2], meta)
        times['postprocess'].append(time.time() - start)

    # 统计
    for key in times:
        times[key] = [t * 1000 for t in times[key]]  # 转换为ms
        avg = np.mean(times[key])
        std = np.std(times[key])
        print(f"{key}: {avg:.1f}±{std:.1f} ms")

    total_avg = sum(np.mean(times[k]) for k in times)
    fps = 1000 / total_avg
    print(f"总计: {total_avg:.1f} ms, {fps:.1f} FPS")
```

---

## 小结

本章介绍了基于Python + RKNNLite的部署实现：

1. **部署方案**：选择Python (简单、快速、可维护)
2. **环境准备**：PC开发环境、RK3588板上环境
3. **推理框架**：完整的预处理→推理→后处理流程
4. **部署脚本**：一键运行、SSH远程部署
5. **网络集成**：JSON序列化、TCP传输
6. **故障排查**：常见问题和性能调试

**关键发现**：
- conf=0.5参数优化实现600×NMS加速
- 416×416分辨率避免NPU Transpose回退
- Python推理延迟基本等同于C++ (差异<10%)

下一章将介绍性能测试与验证。

