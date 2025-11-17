# 第1章 选题依据 - 研究背景分析流程图

```mermaid
flowchart TD
    Start([开始: 选题依据分析]) --> Background[1.1 研究背景及意义]

    Background --> AI_Edge[1.1.1 人工智能与边缘计算发展]
    AI_Edge --> Cloud_Issue{云端计算架构问题}
    Cloud_Issue --> Latency[网络延迟<br/>100-300ms]
    Cloud_Issue --> Bandwidth[带宽成本高<br/>稳定性差]
    Cloud_Issue --> Privacy[数据隐私<br/>安全风险]

    Latency --> Edge_Solution[边缘计算解决方案]
    Bandwidth --> Edge_Solution
    Privacy --> Edge_Solution

    Edge_Solution --> Edge_Benefits{边缘计算优势}
    Edge_Benefits --> Benefit1[毫秒级响应延迟]
    Edge_Benefits --> Benefit2[降低网络带宽消耗]
    Edge_Benefits --> Benefit3[数据本地化处理]

    Background --> Pedestrian[1.1.2 行人检测技术应用价值]
    Pedestrian --> ITS[智能交通系统应用]
    Pedestrian --> AutoDrive[自动驾驶应用]
    Pedestrian --> Surveillance[智能监控应用]

    ITS --> Challenge{技术挑战}
    AutoDrive --> Challenge
    Surveillance --> Challenge

    Challenge --> Scale[尺度变化<br/>10倍以上差异]
    Challenge --> Occlusion[遮挡问题<br/>拥挤场景]
    Challenge --> Environment[复杂环境<br/>光照/天气变化]

    Background --> RK3588[1.1.3 RK3588平台技术优势]
    RK3588 --> CPU[8核异构CPU<br/>4×A76 + 4×A55]
    RK3588 --> NPU[6 TOPS NPU<br/>三核心架构]
    RK3588 --> Video[8K视频编解码<br/>12路摄像头]
    RK3588 --> Interface[丰富外设接口<br/>PCIe/SATA/USB]

    CPU --> Platform_Ready[平台就绪]
    NPU --> Platform_Ready
    Video --> Platform_Ready
    Interface --> Platform_Ready

    Platform_Ready --> Research[1.2 国内外研究现状]

    Research --> DL_Detection[1.2.1 深度学习目标检测算法]
    DL_Detection --> TwoStage[两阶段检测器<br/>R-CNN/Faster R-CNN]
    DL_Detection --> OneStage[单阶段检测器<br/>YOLO/SSD]

    TwoStage --> YOLO_Evolution[YOLO系列演进]
    OneStage --> YOLO_Evolution

    YOLO_Evolution --> YOLOv3[YOLOv3<br/>FPN多尺度]
    YOLO_Evolution --> YOLOv5[YOLOv5<br/>5种规模模型]
    YOLO_Evolution --> YOLOv8[YOLOv8<br/>Anchor-Free设计]

    Research --> Lightweight[1.2.2 模型轻量化技术]
    Lightweight --> Pruning[网络剪枝<br/>50-70%压缩]
    Lightweight --> Distillation[知识蒸馏<br/>大模型→小模型]
    Lightweight --> Quantization[量化技术<br/>INT8/FP16]
    Lightweight --> Efficient_Net[轻量级网络<br/>MobileNet/ShuffleNet]

    Research --> Deployment[1.2.3 边缘AI部署技术]
    Deployment --> Conversion[模型转换<br/>PyTorch→ONNX→RKNN]
    Deployment --> Inference_Engine[推理引擎优化<br/>RKNN-Toolkit2]
    Deployment --> HW_Accel[硬件加速<br/>NPU/RGA]

    Conversion --> Research_Complete[研究现状分析完成]
    Inference_Engine --> Research_Complete
    HW_Accel --> Research_Complete

    Research_Complete --> Content[1.3 研究内容与技术路线]

    Content --> Goals[1.3.1 研究目标]
    Goals --> Goal1[实时性能: ≥20 FPS]
    Goals --> Goal2[检测精度: mAP ≥70%]
    Goals --> Goal3[资源效率: ≤5W功耗]
    Goals --> Goal4[鲁棒性: 多场景稳定]
    Goals --> Goal5[可扩展性: 模块化设计]

    Content --> Tasks[1.3.2 主要研究内容]
    Tasks --> Task1[模型选型与优化<br/>YOLOv5/v8/v7对比]
    Tasks --> Task2[模型转换与量化<br/>RKNN-Toolkit2]
    Tasks --> Task3[推理系统设计<br/>视频→检测→可视化]
    Tasks --> Task4[性能测试与优化<br/>精度+速度验证]

    Content --> Route[1.3.3 技术路线]
    Route --> Phase1[理论研究<br/>1-2周]
    Route --> Phase2[环境搭建<br/>3-4周]
    Route --> Phase3[模型训练<br/>5-7周]
    Route --> Phase4[模型转换<br/>8-9周]
    Route --> Phase5[系统开发<br/>10-11周]
    Route --> Phase6[测试优化<br/>12-13周]
    Route --> Phase7[论文撰写<br/>14-15周]

    Phase1 --> Phase2
    Phase2 --> Phase3
    Phase3 --> Phase4
    Phase4 --> Phase5
    Phase5 --> Phase6
    Phase6 --> Phase7

    Content --> Outcomes[1.3.4 预期成果]
    Outcomes --> Outcome1[完整解决方案<br/>模型+代码+工具链]
    Outcomes --> Outcome2[性能指标验证<br/>mAP/FPS/功耗]
    Outcomes --> Outcome3[技术文档<br/>设计+部署+测试]
    Outcomes --> Outcome4[毕业论文<br/>背景+方案+结果]
    Outcomes --> Outcome5[演示系统<br/>实时检测+可视化]

    Phase7 --> End([第1章完成])
    Outcome5 --> End

    style Start fill:#e1f5e1
    style End fill:#ffe1e1
    style Edge_Solution fill:#fff4e1
    style YOLO_Evolution fill:#e1f0ff
    style Goals fill:#f0e1ff
    style Outcomes fill:#ffe1f0
```

## 流程图说明

### 主要流程

1. **研究背景及意义 (1.1)**
   - 人工智能与边缘计算发展趋势分析
   - 行人检测技术应用价值探讨
   - RK3588平台技术优势总结

2. **国内外研究现状 (1.2)**
   - 深度学习目标检测算法演进（R-CNN → YOLO → YOLOv8）
   - 模型轻量化技术综述（剪枝、蒸馏、量化、轻量级网络）
   - 边缘AI部署技术现状（模型转换、推理优化、硬件加速）

3. **研究内容与技术路线 (1.3)**
   - 5大研究目标（实时性、精度、效率、鲁棒性、可扩展性）
   - 4大研究内容（模型优化、转换部署、系统设计、性能测试）
   - 7阶段技术路线（15周完整计划）
   - 5项预期成果（方案、指标、文档、论文、演示）

### 关键决策点

- **云端 vs 边缘计算**: 延迟、带宽、隐私问题驱动边缘计算选择
- **两阶段 vs 单阶段检测器**: 实时性需求选择YOLO单阶段检测器
- **模型选型**: YOLOv5/v8在精度和速度间取得平衡

### 技术挑战

- 尺度变化（10倍以上）
- 遮挡问题（拥挤场景）
- 复杂环境（光照/天气变化）
- 实时性要求（≥20 FPS）
- 资源约束（≤5W功耗）
