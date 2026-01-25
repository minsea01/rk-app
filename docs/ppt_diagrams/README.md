# 答辩PPT图表集

本目录包含6个PlantUML图，用于毕业答辩PPT。

## 图表列表

| 序号 | 文件名 | 用途 |
|------|--------|------|
| 1 | `01_system_architecture.puml` | 系统整体架构图 - 展示硬件软件分层 |
| 2 | `02_model_conversion_flow.puml` | 模型转换流程图 - PyTorch→ONNX→RKNN |
| 3 | `03_inference_pipeline.puml` | 推理流程图 - 从输入到输出的时序 |
| 4 | `04_data_flow.puml` | 数据流图 - 完整数据流转路径 |
| 5 | `05_network_topology.puml` | 网络拓扑图 - 双千兆网卡架构 |
| 6 | `06_deployment_architecture.puml` | 部署架构图 - 开发到板端部署 |

## 渲染方法

### 方法1: 在线工具 (推荐)

访问 [PlantUML Online Server](https://www.plantuml.com/plantuml/uml/)，粘贴代码即可预览和下载。

### 方法2: VSCode 插件

1. 安装 `PlantUML` 插件 (jebbs.plantuml)
2. 打开 `.puml` 文件
3. `Alt+D` 预览，右键导出 PNG/SVG

### 方法3: 命令行导出

```bash
# 安装 plantuml
sudo apt install plantuml

# 导出单个文件
plantuml -tpng 01_system_architecture.puml

# 批量导出所有图表为PNG
plantuml -tpng *.puml

# 导出为SVG (矢量图，PPT推荐)
plantuml -tsvg *.puml
```

### 方法4: Docker 导出

```bash
docker run --rm -v $(pwd):/data plantuml/plantuml -tpng /data/*.puml
```

## PPT使用建议

1. **推荐格式**: SVG (矢量图，缩放不失真)
2. **分辨率**: PNG导出时建议使用 `-Sdpi=150` 参数
3. **配色**: 图表使用统一的浅色系配色，与PPT模板协调
4. **字体**: 中文使用"微软雅黑"，英文使用默认字体

## 图表内容说明

### 图1: 系统整体架构图
- 硬件层: RK3588 SoC (NPU/CPU/DDR), 双网卡, 存储
- 系统层: Ubuntu, NPU驱动, RGMII驱动
- 运行层: RKNN-Toolkit2-Lite, OpenCV, Python
- 应用层: YOLO推理引擎, 视频流模块, 网络传输

### 图2: 模型转换流程图
- PyTorch阶段: 预训练模型导出
- ONNX阶段: 格式转换与验证
- 量化阶段: 校准数据集准备, INT8量化
- 验证阶段: PC模拟器验证, 精度对比

### 图3: 推理流程图
- 时序图展示从图像输入到检测结果的完整流程
- 标注各阶段延迟: 预处理(2-3ms), NPU推理(20-25ms), 后处理(3-5ms)
- 端到端延迟 <35ms, 帧率 >30 FPS

### 图4: 数据流图
- 输入层: GStreamer采集, 帧缓冲
- 处理层: 色彩转换, 尺寸缩放, 归一化
- 推理层: RKNN Runtime, 3核NPU并行
- 后处理层: DFL解码, NMS过滤, 坐标变换
- 输出层: 序列化, 可视化, 网络发送

### 图5: 网络拓扑图
- eth0 (RGMII): 相机输入网口, 192.168.1.100
- eth1 (RGMII): 数据上传网口, 192.168.2.100
- 双网卡隔离设计，保障带宽和稳定性
- 吞吐量指标: ≥900 Mbps

### 图6: 部署架构图
- 开发环境: WSL2 Ubuntu 22.04
- 模型开发: PyTorch训练, ONNX导出, RKNN转换
- 交叉编译: CMake + GCC Toolchain
- 目标平台: RK3588开发板
- 部署模式: Python Runner / C++ CLI
