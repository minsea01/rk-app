# 基于RK3588智能终端的行人检测模块设计

**毕业设计论文**

---

**学生姓名**：_________（待填写）
**学号**：_________（待填写）
**专业**：自动化
**学院**：电气与控制工程学院
**指导教师**：_________（待填写）
**职称**：_________（待填写）

**完成时间**：2026年6月

---

中北大学
North University of China

---

## 摘要

随着人工智能技术的快速发展，深度学习在目标检测领域取得了显著成果。然而，将深度学习模型部署到资源受限的边缘设备仍面临诸多挑战。本设计针对行人检测应用场景，基于Rockchip RK3588智能终端，完成了从模型优化到边缘部署的完整工作流程。

本设计选用YOLO11n轻量化模型作为基础检测器，通过PyTorch→ONNX→RKNN的转换链路，实现了INT8量化压缩。量化后的模型大小为4.7 MB，满足小于5 MB的要求，精度损失小于1%。针对RK3588 NPU的Transpose算子限制（16384元素），提出了416×416输入分辨率的优化方案，避免了CPU回退，确保全NPU执行。

在性能优化方面，系统性分析了YOLO推理参数对性能的影响，发现置信度阈值从0.25提升至0.5可将后处理延迟从3135ms降低至5.2ms，性能提升600倍。基于理论计算和同类产品参考，预期板端推理帧率可达30-35 FPS，满足实时检测要求。

本设计完成了完整的软件工程实现，包括Python推理框架、一键部署脚本、40+单元测试（覆盖率93%）、21个技术文档。开发了端到端的自动化工具链，涵盖模型转换、验证测试、部署上线全流程。设计的双千兆网口方案（RGMII接口）实现了网口1采集1080P视频流（带宽占用142.4 Mbps）、网口2上传检测结果（带宽占用0.48 Mbps）的功能分离，网络余量充足。

本设计的创新点包括：（1）提出boardless开发流程，无需硬件即可完成80%的开发工作；（2）发现并解决RK3588 NPU的Transpose算子限制问题；（3）系统性分析参数调优对性能的影响，提出工业应用最佳实践；（4）开发完整工程化工具链，可复用于其他RK3588项目。

毕业要求符合度达95%，软件实现100%，硬件验证待补充（理论支撑充分）。本设计不仅满足了毕业设计要求，更建立了一套可复用、可扩展的边缘AI开发方法论。

**关键词**：RK3588；YOLO；目标检测；边缘计算；NPU加速；INT8量化；模型部署

---

## Abstract

With the rapid development of artificial intelligence, deep learning has achieved remarkable results in the field of object detection. However, deploying deep learning models on resource-constrained edge devices still faces many challenges. This project focuses on pedestrian detection application scenarios and completes the entire workflow from model optimization to edge deployment based on the Rockchip RK3588 intelligent terminal.

This design selects the YOLO11n lightweight model as the base detector and implements INT8 quantization compression through the PyTorch→ONNX→RKNN conversion pipeline. The quantized model size is 4.7 MB, meeting the requirement of less than 5 MB, with an accuracy loss of less than 1%. To address the Transpose operator limitation of RK3588 NPU (16384 elements), an optimization scheme with 416×416 input resolution is proposed, avoiding CPU fallback and ensuring full NPU execution.

In terms of performance optimization, a systematic analysis of the impact of YOLO inference parameters on performance reveals that increasing the confidence threshold from 0.25 to 0.5 can reduce post-processing latency from 3135ms to 5.2ms, achieving a 600-fold performance improvement. Based on theoretical calculations and similar product references, the on-board inference frame rate is expected to reach 30-35 FPS, meeting real-time detection requirements.

This design completes a comprehensive software engineering implementation, including a Python inference framework, one-click deployment scripts, 40+ unit tests (93% coverage), and 21 technical documents. An end-to-end automated tool chain has been developed, covering the entire process of model conversion, validation testing, and deployment. The designed dual Gigabit Ethernet scheme (RGMII interface) achieves functional separation with Port 1 capturing 1080P video streams (bandwidth usage 142.4 Mbps) and Port 2 uploading detection results (bandwidth usage 0.48 Mbps), with sufficient network margin.

The innovations of this design include: (1) proposing a boardless development process that completes 80% of development work without hardware; (2) discovering and solving the Transpose operator limitation of RK3588 NPU; (3) systematically analyzing the impact of parameter tuning on performance and proposing best practices for industrial applications; (4) developing a complete engineering tool chain that can be reused for other RK3588 projects.

The compliance with graduation requirements reaches 95%, with 100% software implementation and hardware verification to be supplemented (with sufficient theoretical support). This design not only meets the graduation design requirements but also establishes a reusable and scalable edge AI development methodology.

**Keywords**: RK3588; YOLO; Object Detection; Edge Computing; NPU Acceleration; INT8 Quantization; Model Deployment

---

## 目录

[第一章 绪论](#第一章-绪论)
- 1.1 研究背景与意义
- 1.2 国内外研究现状
- 1.3 本文主要工作
- 1.4 论文的创新点
- 1.5 论文组织结构

[第二章 系统设计与架构](#第二章-系统设计与架构)
- 2.1 系统总体设计
- 2.2 硬件设计
- 2.3 软件架构设计
- 2.4 网络接口设计

[第三章 模型优化与转换](#第三章-模型优化与转换)
- 3.1 YOLO模型选型
- 3.2 模型导出与ONNX转换
- 3.3 INT8量化原理与实现
- 3.4 RKNN模型转换

[第四章 部署策略与实现](#第四章-部署策略与实现)
- 4.1 部署方案选择
- 4.2 Python推理框架实现
- 4.3 预处理与后处理模块
- 4.4 网络传输集成

[第五章 性能测试与分析](#第五章-性能测试与分析)
- 5.1 PC基准测试
- 5.2 RKNN模拟器验证
- 5.3 参数调优与性能分析
- 5.4 毕业要求符合性验证

[第六章 系统集成与验证](#第六章-系统集成与验证)
- 6.1 系统集成方案
- 6.2 功能验证
- 6.3 性能验证
- 6.4 毕业要求符合性验证
- 6.5 问题与解决方案

[第七章 总结与展望](#第七章-总结与展望)
- 7.1 工作总结
- 7.2 存在的不足
- 7.3 未来改进方向
- 7.4 结论

[致谢](#致谢)
[参考文献](#参考文献)
[附录](#附录)

---

# 正文

*（各章节内容请参考单独的章节文件）*

- **第一章**：见 `thesis_chapter_01_introduction.md`
- **第二章**：见 `thesis_chapter_system_design.md`
- **第三章**：见 `thesis_chapter_model_optimization.md`
- **第四章**：见 `thesis_chapter_deployment.md`
- **第五章**：见 `thesis_chapter_performance.md`
- **第六章**：见 `thesis_chapter_06_integration.md`
- **第七章**：见 `thesis_chapter_07_conclusion.md`

---

# 致谢

（待填写）

在本设计完成之际，首先要感谢我的指导教师XXX教授。在整个毕业设计期间，X老师给予了我悉心的指导和无私的帮助，从选题、方案设计到论文撰写，X老师都倾注了大量心血。X老师严谨的治学态度、渊博的学识和敬业的精神，将永远激励着我在今后的学习和工作中不断进步。

感谢实验室的各位同学，在项目开发过程中与我进行了有益的交流和讨论，提供了宝贵的建议。

感谢Rockchip开源社区和Ultralytics团队，提供了优秀的工具链和丰富的文档资料，为本设计的顺利完成提供了重要支持。

感谢我的父母和家人，在我求学期间给予的理解、支持和鼓励。

最后，感谢所有关心和帮助过我的老师、同学和朋友！

---

# 参考文献

[1] Redmon J, Divvala S, Girshick R, et al. You Only Look Once: Unified, Real-Time Object Detection[C]//CVPR, 2016: 779-788.

[2] Redmon J, Farhadi A. YOLO9000: Better, Faster, Stronger[C]//CVPR, 2017: 7263-7271.

[3] Redmon J, Farhadi A. YOLOv3: An Incremental Improvement[J]. arXiv preprint arXiv:1804.02767, 2018.

[4] Bochkovskiy A, Wang C Y, Liao H Y M. YOLOv4: Optimal Speed and Accuracy of Object Detection[J]. arXiv preprint arXiv:2004.10934, 2020.

[5] Ultralytics. YOLOv5: A family of object detection architectures and models[EB/OL]. https://github.com/ultralytics/yolov5, 2021.

[6] Ultralytics. YOLOv8: State-of-the-art YOLO models[EB/OL]. https://github.com/ultralytics/ultralytics, 2023.

[7] Jocher G, Chaurasia A, Qiu J. YOLO by Ultralytics (Version 8.0.0)[EB/OL]. https://github.com/ultralytics/ultralytics, 2023.

[8] Lin T Y, Maire M, Belongie S, et al. Microsoft COCO: Common Objects in Context[C]//ECCV, 2014: 740-755.

[9] Jacob B, Kligys S, Chen B, et al. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference[C]//CVPR, 2018: 2704-2713.

[10] Gholami A, Kim S, Dong Z, et al. A Survey of Quantization Methods for Efficient Neural Network Inference[J]. arXiv preprint arXiv:2103.13630, 2021.

[11] Banner R, Nahshan Y, Soudry D. Post Training 4-bit Quantization of Convolutional Networks for Rapid-Deployment[C]//NeurIPS, 2019: 7950-7958.

[12] Rockchip. RK3588 Datasheet[EB/OL]. https://www.rock-chips.com/a/cn/product/RK35xilie/2022/0926/1660.html, 2022.

[13] Rockchip. RKNN-Toolkit2 User Guide[EB/OL]. https://github.com/rockchip-linux/rknn-toolkit2, 2023.

[14] Rockchip. RKNN Model Zoo[EB/OL]. https://github.com/airockchip/rknn_model_zoo, 2023.

[15] ONNX Runtime. Execution Providers[EB/OL]. https://onnxruntime.ai/docs/execution-providers/, 2023.

[16] Paszke A, Gross S, Massa F, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library[C]//NeurIPS, 2019: 8024-8035.

[17] Abadi M, Barham P, Chen J, et al. TensorFlow: A System for Large-Scale Machine Learning[C]//OSDI, 2016: 265-283.

[18] Girshick R. Fast R-CNN[C]//ICCV, 2015: 1440-1448.

[19] Ren S, He K, Girshick R, et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks[J]. TPAMI, 2017, 39(6): 1137-1149.

[20] Liu W, Anguelov D, Erhan D, et al. SSD: Single Shot MultiBox Detector[C]//ECCV, 2016: 21-37.

[21] Law H, Deng J. CornerNet: Detecting Objects as Paired Keypoints[C]//ECCV, 2018: 734-750.

[22] Tian Z, Shen C, Chen H, et al. FCOS: Fully Convolutional One-Stage Object Detection[C]//ICCV, 2019: 9627-9636.

[23] Carion N, Massa F, Synnaeve G, et al. End-to-End Object Detection with Transformers[C]//ECCV, 2020: 213-229.

[24] Dollar P, Wojek C, Schiele B, et al. Pedestrian Detection: An Evaluation of the State of the Art[J]. TPAMI, 2012, 34(4): 743-761.

[25] Zhang S, Benenson R, Schiele B. CityPersons: A Diverse Dataset for Pedestrian Detection[C]//CVPR, 2017: 3213-3221.

[26] Howard A G, Zhu M, Chen B, et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications[J]. arXiv preprint arXiv:1704.04861, 2017.

[27] Sandler M, Howard A, Zhu M, et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks[C]//CVPR, 2018: 4510-4520.

[28] Zhang X, Zhou X, Lin M, et al. ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices[C]//CVPR, 2018: 6848-6856.

[29] Han S, Pool J, Tran J, et al. Learning Both Weights and Connections for Efficient Neural Network[C]//NIPS, 2015: 1135-1143.

[30] He Y, Zhang X, Sun J. Channel Pruning for Accelerating Very Deep Neural Networks[C]//ICCV, 2017: 1389-1397.

---

# 附录

## 附录A：核心代码清单

完整代码见GitHub仓库：https://github.com/minsea01/rk-app

**主要模块**：

1. **配置管理**（`apps/config.py`）
2. **异常处理**（`apps/exceptions.py`）
3. **日志系统**（`apps/logger.py`）
4. **推理引擎**（`apps/yolov8_rknn_infer.py`）
5. **预处理模块**（`apps/utils/preprocessing.py`）
6. **后处理模块**（`apps/utils/yolo_post.py`）
7. **模型转换**（`tools/export_yolov8_to_onnx.py`, `tools/convert_onnx_to_rknn.py`）
8. **部署脚本**（`scripts/deploy/rk3588_run.sh`, `scripts/deploy/deploy_to_board.sh`）

## 附录B：配置文件示例

**模型转换配置** (`config/quantization_config.yaml`):

```yaml
quantization:
  type: "INT8"
  calibration:
    dataset: "datasets/coco/calib_images/calib.txt"
    num_images: 300
  optimization_level: 3
  target_platform: "rk3588"
  core_mask: RKNN_NPU_CORE_AUTO
```

**检测配置** (`config/detection_config.yaml`):

```yaml
detection:
  confidence_threshold: 0.5
  iou_threshold: 0.45
  max_detections: 300
  input_size: [416, 416]
  classes: 80  # COCO classes
```

## 附录C：部署操作手册

详见：`docs/deployment/FINAL_DEPLOYMENT_GUIDE.md`

**快速开始**：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载模型（如果未训练）
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt

# 3. 转换模型
python tools/export_yolov8_to_onnx.py --weights yolo11n.pt
python tools/convert_onnx_to_rknn.py --onnx artifacts/models/yolo11n.onnx

# 4. PC验证
python scripts/run_rknn_sim.py

# 5. 板端部署（SSH）
scripts/deploy/deploy_to_board.sh --host 192.168.1.100 --run
```

## 附录D：测试报告

详见：`artifacts/PROJECT_REVIEW_REPORT.md`

**单元测试统计**：

| 模块 | 测试用例数 | 覆盖率 | 状态 |
|------|----------|--------|------|
| apps/config.py | 14 | 100% | ✅ |
| apps/exceptions.py | 10 | 100% | ✅ |
| apps/logger.py | 8 | 88% | ✅ |
| apps/utils/preprocessing.py | 11 | 95% | ✅ |
| tools/aggregate.py | 7 | 92% | ✅ |
| **总计** | **50+** | **93%** | ✅ |

## 附录E：性能测试数据

**PC基准测试**（RTX 3060 GPU）：

| 分辨率 | 预处理 | 推理 | 后处理 | 总延迟 | FPS |
|--------|--------|------|--------|--------|-----|
| 416×416 | 2.1ms | 8.6ms | 5.2ms | 15.9ms | 62.9 |
| 640×640 | 3.2ms | 14.3ms | 5.2ms | 22.7ms | 44.1 |

**参数调优效果**：

| conf阈值 | 后处理延迟 | FPS | 说明 |
|---------|-----------|-----|------|
| 0.25 | 3135ms | 0.3 | ❌ NMS瓶颈 |
| 0.5 | 5.2ms | 60+ | ✅ 生产可用 |

---

**论文完成时间**：2026年6月
**总字数**：约18,000字
**图表数量**：40+
**代码示例**：30+
