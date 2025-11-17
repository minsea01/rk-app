# 完整技术工作流程图

```mermaid
flowchart TD
    Start([项目启动]) --> Research[阶段1: 理论研究<br/>Week 1-2]

    %% 阶段1: 理论研究
    Research --> Literature[文献调研]
    Literature --> Paper1[深度学习基础<br/>CNN/目标检测]
    Literature --> Paper2[YOLO系列算法<br/>v3/v5/v8对比]
    Literature --> Paper3[模型轻量化技术<br/>剪枝/蒸馏/量化]
    Literature --> Paper4[边缘AI部署<br/>RKNN工具链]

    Paper1 --> Platform_Study[RK3588平台调研]
    Paper2 --> Platform_Study
    Paper3 --> Platform_Study
    Paper4 --> Platform_Study

    Platform_Study --> HW_Spec[硬件规格<br/>8核CPU+6TOPS NPU]
    Platform_Study --> SW_Stack[软件栈<br/>RKNN-Toolkit2]
    Platform_Study --> Cases[应用案例<br/>智能监控/自动驾驶]

    HW_Spec --> Report1[撰写开题报告]
    SW_Stack --> Report1
    Cases --> Report1

    Report1 --> Setup[阶段2: 环境搭建<br/>Week 3-4]

    %% 阶段2: 环境搭建
    Setup --> PC_Env[PC端开发环境]
    PC_Env --> Install_PyTorch[安装PyTorch 2.0+<br/>CUDA 11.7]
    PC_Env --> Install_RKNN[安装RKNN-Toolkit2<br/>模型转换工具]
    PC_Env --> Install_OpenCV[安装OpenCV 4.9<br/>图像处理]

    Setup --> Board_Env[RK3588开发板环境]
    Board_Env --> Flash_OS[烧录Ubuntu 22.04<br/>ARM64系统]
    Board_Env --> Install_Runtime[安装RKNN Runtime<br/>推理库]
    Board_Env --> Camera_Test[摄像头测试<br/>USB/MIPI-CSI]

    Setup --> Dataset_Prep[数据集准备]
    Dataset_Prep --> Download_COCO[下载COCO数据集<br/>20万+图像]
    Dataset_Prep --> Download_City[下载CityPersons<br/>5000张城市场景]
    Dataset_Prep --> Filter_Person[筛选行人类别<br/>category_id=1]
    Dataset_Prep --> Split_Data[划分训练/验证集<br/>80%/20%]

    Install_RKNN --> Train[阶段3: 模型训练<br/>Week 5-7]
    Flash_OS --> Train
    Split_Data --> Train

    %% 阶段3: 模型训练
    Train --> Select_Model{选择基础模型}
    Select_Model --> Try_YOLOv5[YOLOv5s<br/>7.2M参数]
    Select_Model --> Try_YOLOv8[YOLOv8n<br/>3.2M参数]

    Try_YOLOv5 --> Baseline_Train[基线模型训练]
    Try_YOLOv8 --> Baseline_Train

    Baseline_Train --> Load_Pretrain[加载COCO预训练权重<br/>迁移学习]
    Load_Pretrain --> Config_Hyper[配置超参数<br/>lr=0.01, batch=16]
    Config_Hyper --> Data_Aug[数据增强<br/>翻转/缩放/色彩抖动]

    Data_Aug --> GPU_Train[GPU训练<br/>RTX 3060, 50 epochs]
    GPU_Train --> Eval_Baseline[评估基线精度<br/>mAP@0.5]

    Eval_Baseline --> Baseline_OK{精度满足要求?}
    Baseline_OK -->|否| Adjust_Hyper[调整超参数<br/>增加训练轮次]
    Adjust_Hyper --> GPU_Train

    Baseline_OK -->|是| Optimize[模型优化]

    Optimize --> Pruning[网络剪枝]
    Pruning --> Analyze_Layer[分析层重要性<br/>L1-norm/BN参数]
    Analyze_Layer --> Remove_Channel[移除冗余通道<br/>保留50-60%]
    Remove_Channel --> Finetune1[微调剪枝模型<br/>恢复精度]

    Optimize --> KD[知识蒸馏<br/>可选]
    KD --> Teacher_Model[教师模型<br/>YOLOv8l/x大模型]
    KD --> Student_Model[学生模型<br/>YOLOv8n小模型]
    Teacher_Model --> Distill_Loss[蒸馏损失<br/>软标签+硬标签]
    Student_Model --> Distill_Loss
    Distill_Loss --> Finetune2[微调学生模型<br/>学习教师知识]

    Finetune1 --> Final_Model[最终优化模型]
    Finetune2 --> Final_Model

    Final_Model --> Eval_Final[评估最终精度<br/>mAP@0.5 ≥70%]

    Eval_Final --> Precision_OK{精度达标?}
    Precision_OK -->|否| Back_Optimize[返回优化阶段<br/>调整策略]
    Back_Optimize --> Optimize

    Precision_OK -->|是| Convert[阶段4: 模型转换<br/>Week 8-9]

    %% 阶段4: 模型转换
    Convert --> Export_ONNX[导出ONNX格式]
    Export_ONNX --> Set_Opset[设置opset=12<br/>兼容性最佳]
    Set_Opset --> Simplify[onnx-simplifier优化<br/>常量折叠+算子融合]

    Simplify --> Verify_ONNX[验证ONNX模型]
    Verify_ONNX --> ONNX_Inference[ONNXRuntime推理<br/>GPU加速]
    ONNX_Inference --> Compare_Acc[对比PyTorch精度<br/>误差<0.1%]

    Compare_Acc --> ONNX_OK{ONNX精度正常?}
    ONNX_OK -->|否| Fix_Export[修复导出问题<br/>检查算子支持]
    Fix_Export --> Export_ONNX

    ONNX_OK -->|是| RKNN_Convert[RKNN模型转换]

    RKNN_Convert --> Prepare_Calib[准备校准数据集]
    Prepare_Calib --> Select_Images[选择500-1000张<br/>代表性图像]
    Select_Images --> Abs_Path[生成绝对路径列表<br/>find + realpath]

    Abs_Path --> Config_RKNN[配置RKNN-Toolkit2]
    Config_RKNN --> Set_Target[设置目标平台<br/>target_platform='rk3588']
    Set_Target --> Set_Quant[设置量化参数<br/>do_quantization=True]
    Set_Quant --> Set_Dtype[设置数据类型<br/>dtype='w8a8' INT8]

    Set_Dtype --> Build_RKNN[构建RKNN模型]
    Build_RKNN --> Load_ONNX_Model[加载ONNX模型]
    Load_ONNX_Model --> Build_Graph[图优化<br/>算子融合+重排]
    Build_Graph --> Quantization[INT8量化<br/>统计激活值分布]

    Quantization --> Export_RKNN[导出.rknn文件]

    Export_RKNN --> Validate[PC模拟器验证]
    Validate --> Sim_Config[配置模拟器<br/>data_format='nhwc']
    Sim_Config --> Sim_Inference[模拟器推理<br/>RKNN-Toolkit2]
    Sim_Inference --> Compare_ONNX[对比ONNX精度<br/>计算mAP差异]

    Compare_ONNX --> Quant_OK{量化精度损失≤2%?}
    Quant_OK -->|否| Adjust_Quant[调整量化策略<br/>混合精度/更多校准数据]
    Adjust_Quant --> Config_RKNN

    Quant_OK -->|是| Transfer_Board[传输到RK3588<br/>scp/SSH]

    Transfer_Board --> Develop[阶段5: 系统开发<br/>Week 10-11]

    %% 阶段5: 系统开发
    Develop --> Design_Arch[系统架构设计]
    Design_Arch --> Module_Diagram[模块划分<br/>5大模块]
    Module_Diagram --> Thread_Design[线程设计<br/>采集/推理/后处理]
    Thread_Design --> Interface_Design[接口设计<br/>模块间数据流]

    Interface_Design --> Implement[代码实现]

    Implement --> Video_Module[视频采集模块<br/>C++/Python]
    Video_Module --> V4L2_USB[USB摄像头<br/>V4L2接口]
    Video_Module --> MIPI_CSI[MIPI-CSI摄像头<br/>专用驱动]
    Video_Module --> Thread_Capture[独立线程<br/>生产者模式]

    Implement --> Preprocess_Module[预处理模块]
    Preprocess_Module --> Color_CVT[BGR→RGB转换<br/>OpenCV]
    Preprocess_Module --> Resize_Image[缩放到640×640<br/>或416×416]
    Preprocess_Module --> To_NHWC[NHWC格式<br/>(1,H,W,3)]
    Preprocess_Module --> RGA_Opt[RGA硬件加速<br/>可选]

    Implement --> Inference_Module[推理模块<br/>核心]
    Inference_Module --> Init_RKNN[初始化RKNN<br/>rknn_init()]
    Inference_Module --> Load_Model[加载.rknn模型<br/>rknn_load_rknn()]
    Inference_Module --> Run_Inference[执行推理<br/>rknn_run()]
    Inference_Module --> Get_Output[获取输出<br/>rknn_outputs_get()]

    Implement --> Postprocess_Module[后处理模块]
    Postprocess_Module --> Decode_Box[解码边界框<br/>cx,cy,w,h→x1,y1,x2,y2]
    Postprocess_Module --> Filter_Conf[置信度过滤<br/>threshold≥0.5]
    Postprocess_Module --> NMS_Algo[NMS算法<br/>IoU threshold=0.45]
    Postprocess_Module --> Scale_Coords[坐标缩放<br/>映射回原图]
    Postprocess_Module --> SIMD_Opt[SIMD优化<br/>向量化计算]

    Implement --> Visual_Module[可视化模块]
    Visual_Module --> Draw_Rect[绘制矩形框<br/>cv2.rectangle()]
    Visual_Module --> Put_Text[标注文字<br/>类别+置信度]
    Visual_Module --> Display_HDMI[HDMI输出<br/>cv2.imshow()]
    Visual_Module --> Save_Result[保存结果<br/>图像/视频/JSON]

    Thread_Capture --> Integration[模块集成]
    SIMD_Opt --> Integration
    Save_Result --> Integration

    Integration --> Build_App[编译应用程序]
    Build_App --> CMake_Config[CMake配置<br/>链接RKNN库]
    CMake_Config --> Cross_Compile[交叉编译<br/>aarch64-linux-gnu-gcc]
    Cross_Compile --> Deploy_Board[部署到开发板<br/>./rk3588_run.sh]

    Deploy_Board --> Test[阶段6: 测试优化<br/>Week 12-13]

    %% 阶段6: 测试优化
    Test --> Func_Test[功能测试]
    Func_Test --> Test_Single[单张图像测试<br/>验证检测结果]
    Func_Test --> Test_Video[视频流测试<br/>实时处理]
    Func_Test --> Test_Multi[多场景测试<br/>室内/室外/夜间]

    Test --> Perf_Test[性能测试]
    Perf_Test --> Measure_FPS[测量FPS<br/>帧率统计]
    Perf_Test --> Measure_Latency[测量延迟<br/>端到端时间]
    Perf_Test --> Measure_Power[测量功耗<br/>NPU工作功率]
    Perf_Test --> Measure_Accuracy[测量精度<br/>mAP@0.5]

    Test --> Stress_Test[压力测试]
    Stress_Test --> Long_Run[长时间运行<br/>8小时稳定性]
    Stress_Test --> High_Load[高负载测试<br/>多路视频]
    Stress_Test --> Memory_Leak[内存泄漏检测<br/>valgrind]

    Measure_Accuracy --> Analyze_Result[分析测试结果]
    Long_Run --> Analyze_Result

    Analyze_Result --> Meet_Target{达到目标?<br/>FPS≥20, mAP≥70%, 功耗≤5W}

    Meet_Target -->|否| Optimize_System[系统优化]

    Optimize_System --> Opt_Strategy{优化策略}
    Opt_Strategy --> Opt_Model[模型优化<br/>降低分辨率/简化网络]
    Opt_Strategy --> Opt_Code[代码优化<br/>减少拷贝/并行化]
    Opt_Strategy --> Opt_Param[参数优化<br/>调整阈值/NMS参数]

    Opt_Model --> Retest[重新测试]
    Opt_Code --> Retest
    Opt_Param --> Retest

    Retest --> Analyze_Result

    Meet_Target -->|是| Document[阶段7: 论文撰写<br/>Week 14-15]

    %% 阶段7: 论文撰写
    Document --> Collect_Data[整理实验数据]
    Collect_Data --> Results_Table[性能指标表格<br/>FPS/mAP/功耗]
    Collect_Data --> Comparison_Table[对比分析表格<br/>vs其他方法]
    Collect_Data --> Charts[可视化图表<br/>精度曲线/速度对比]

    Results_Table --> Write_Chapters[撰写论文章节]

    Write_Chapters --> Chapter1[第1章: 绪论<br/>背景+意义+现状]
    Write_Chapters --> Chapter2[第2章: 系统设计<br/>架构+模块设计]
    Write_Chapters --> Chapter3[第3章: 模型优化<br/>选型+量化+转换]
    Write_Chapters --> Chapter4[第4章: 系统实现<br/>代码实现+部署]
    Write_Chapters --> Chapter5[第5章: 测试分析<br/>性能测试+结果分析]
    Write_Chapters --> Chapter6[第6章: 总结展望<br/>成果+不足+未来]

    Chapter1 --> Review_Draft[论文初稿审阅]
    Chapter2 --> Review_Draft
    Chapter3 --> Review_Draft
    Chapter4 --> Review_Draft
    Chapter5 --> Review_Draft
    Chapter6 --> Review_Draft

    Review_Draft --> Revise{需要修改?}
    Revise -->|是| Modify[修改论文<br/>导师意见]
    Modify --> Review_Draft

    Revise -->|否| Final_Draft[论文定稿]

    Final_Draft --> Defense_Prep[准备答辩材料]
    Defense_Prep --> PPT[制作PPT<br/>20-30页]
    Defense_Prep --> Demo_Video[录制演示视频<br/>5-10分钟]
    Defense_Prep --> QA_Prep[准备答辩问题<br/>技术细节]

    PPT --> End([项目完成<br/>准备答辩])
    Demo_Video --> End
    QA_Prep --> End

    style Start fill:#e1f5e1
    style End fill:#ffe1e1
    style Research fill:#e1f0ff
    style Setup fill:#fff4e1
    style Train fill:#f0e1ff
    style Convert fill:#ffe1f0
    style Develop fill:#e1ffe1
    style Test fill:#ffe1e1
    style Document fill:#e1f5e1
```

## 完整工作流程说明

### 阶段1: 理论研究（Week 1-2）

**目标**: 建立坚实的理论基础，明确技术路线

**主要任务**:
1. **文献调研**
   - 深度学习基础（CNN、目标检测算法）
   - YOLO系列算法演进（v3/v5/v8对比分析）
   - 模型轻量化技术（剪枝、蒸馏、量化）
   - 边缘AI部署技术（RKNN工具链）

2. **平台调研**
   - RK3588硬件规格（8核CPU + 6 TOPS NPU）
   - 软件开发栈（RKNN-Toolkit2工具链）
   - 成功应用案例（智能监控、自动驾驶）

3. **输出**: 开题报告

---

### 阶段2: 环境搭建（Week 3-4）

**目标**: 搭建完整的开发、训练、部署环境

**主要任务**:
1. **PC端开发环境**
   - PyTorch 2.0+ (CUDA 11.7)
   - RKNN-Toolkit2（模型转换工具）
   - OpenCV 4.9（图像处理）

2. **RK3588开发板环境**
   - 烧录Ubuntu 22.04 ARM64系统
   - 安装RKNN Runtime推理库
   - 摄像头测试（USB/MIPI-CSI）

3. **数据集准备**
   - 下载COCO数据集（20万+图像）
   - 下载CityPersons数据集（5000张城市场景）
   - 筛选行人类别（category_id=1）
   - 划分训练/验证集（80%/20%）

---

### 阶段3: 模型训练（Week 5-7）

**目标**: 训练并优化高精度、轻量级的行人检测模型

**主要任务**:
1. **基线模型训练**
   - 选择YOLOv5s或YOLOv8n作为基础架构
   - 加载COCO预训练权重（迁移学习）
   - 配置超参数（lr=0.01, batch=16）
   - 数据增强（翻转、缩放、色彩抖动）
   - GPU训练（RTX 3060, 50 epochs）
   - 评估基线精度（mAP@0.5）

2. **模型优化**
   - **网络剪枝**:
     - 分析层重要性（L1-norm、BN参数）
     - 移除冗余通道（保留50-60%）
     - 微调剪枝模型（恢复精度）

   - **知识蒸馏**（可选）:
     - 教师模型: YOLOv8l/x大模型
     - 学生模型: YOLOv8n小模型
     - 蒸馏损失: 软标签 + 硬标签
     - 微调学生模型（学习教师知识）

3. **精度验证**: 最终mAP@0.5 ≥70%

---

### 阶段4: 模型转换（Week 8-9）

**目标**: 将PyTorch模型转换为RKNN格式并完成INT8量化

**主要任务**:
1. **导出ONNX格式**
   - 设置opset=12（兼容性最佳）
   - onnx-simplifier优化（常量折叠、算子融合）
   - 验证ONNX模型精度（对比PyTorch，误差<0.1%）

2. **RKNN模型转换**
   - **准备校准数据集**:
     - 选择500-1000张代表性图像
     - 生成绝对路径列表（`find ... -exec realpath {} \;`）

   - **配置RKNN-Toolkit2**:
     - `target_platform='rk3588'`
     - `do_quantization=True`
     - `dtype='w8a8'` (INT8量化)

   - **构建RKNN模型**:
     - 加载ONNX模型
     - 图优化（算子融合、重排）
     - INT8量化（统计激活值分布）
     - 导出.rknn文件

3. **PC模拟器验证**
   - 配置模拟器（`data_format='nhwc'`）
   - 模拟器推理
   - 对比ONNX精度（量化精度损失≤2%）

4. **传输到RK3588**: scp/SSH部署

---

### 阶段5: 系统开发（Week 10-11）

**目标**: 开发完整的边缘端实时推理应用程序

**主要任务**:
1. **系统架构设计**
   - 模块划分（5大模块）
   - 线程设计（采集、推理、后处理独立线程）
   - 接口设计（模块间数据流）

2. **代码实现**
   - **视频采集模块**:
     - USB摄像头（V4L2接口）
     - MIPI-CSI摄像头（专用驱动）
     - 独立线程（生产者模式）

   - **预处理模块**:
     - BGR→RGB转换
     - 缩放到640×640或416×416
     - NHWC格式转换 (1,H,W,3)
     - RGA硬件加速（可选）

   - **推理模块**（核心）:
     - 初始化RKNN (`rknn_init()`)
     - 加载.rknn模型 (`rknn_load_rknn()`)
     - 执行推理 (`rknn_run()`)
     - 获取输出 (`rknn_outputs_get()`)

   - **后处理模块**:
     - 解码边界框（cx,cy,w,h → x1,y1,x2,y2）
     - 置信度过滤（threshold≥0.5）
     - NMS算法（IoU threshold=0.45）
     - 坐标缩放（映射回原图）
     - SIMD优化（向量化计算）

   - **可视化模块**:
     - 绘制矩形框 (`cv2.rectangle()`)
     - 标注文字（类别+置信度）
     - HDMI输出 (`cv2.imshow()`)
     - 保存结果（图像/视频/JSON）

3. **模块集成**
   - CMake配置（链接RKNN库）
   - 交叉编译（aarch64-linux-gnu-gcc）
   - 部署到开发板（`./rk3588_run.sh`）

---

### 阶段6: 测试优化（Week 12-13）

**目标**: 全面测试系统性能，优化至满足设计目标

**主要任务**:
1. **功能测试**
   - 单张图像测试（验证检测结果）
   - 视频流测试（实时处理）
   - 多场景测试（室内/室外/夜间）

2. **性能测试**
   - 测量FPS（帧率统计）
   - 测量延迟（端到端时间）
   - 测量功耗（NPU工作功率）
   - 测量精度（mAP@0.5）

3. **压力测试**
   - 长时间运行（8小时稳定性）
   - 高负载测试（多路视频）
   - 内存泄漏检测（valgrind）

4. **性能优化**（如未达到目标: FPS≥20, mAP≥70%, 功耗≤5W）
   - **模型优化**: 降低分辨率（640→416）、简化网络
   - **代码优化**: 减少数据拷贝、流水线并行化
   - **参数优化**: 调整置信度阈值、NMS参数

---

### 阶段7: 论文撰写（Week 14-15）

**目标**: 撰写高质量毕业论文，准备答辩材料

**主要任务**:
1. **整理实验数据**
   - 性能指标表格（FPS/mAP/功耗）
   - 对比分析表格（vs其他方法）
   - 可视化图表（精度曲线、速度对比）

2. **撰写论文章节**
   - **第1章**: 绪论（研究背景、意义、国内外研究现状）
   - **第2章**: 系统设计（总体架构、模块设计）
   - **第3章**: 模型优化（模型选型、量化技术、转换流程）
   - **第4章**: 系统实现（代码实现、部署方案）
   - **第5章**: 测试分析（性能测试、结果分析）
   - **第6章**: 总结展望（研究成果、存在不足、未来工作）

3. **论文审阅与修改**
   - 导师审阅
   - 根据意见修改
   - 定稿

4. **准备答辩材料**
   - 制作PPT（20-30页）
   - 录制演示视频（5-10分钟）
   - 准备答辩问题（技术细节、创新点）

---

## 关键里程碑

| 阶段 | 周次 | 关键输出 | 验收标准 |
|-----|------|---------|---------|
| 理论研究 | 1-2 | 开题报告 | 导师审批通过 |
| 环境搭建 | 3-4 | 开发环境+数据集 | 环境可用，数据集准备完成 |
| 模型训练 | 5-7 | 优化后的PyTorch模型 | mAP@0.5 ≥70% |
| 模型转换 | 8-9 | RKNN量化模型 | 量化精度损失≤2% |
| 系统开发 | 10-11 | 实时推理应用程序 | 程序可在RK3588上运行 |
| 测试优化 | 12-13 | 性能测试报告 | FPS≥20, mAP≥70%, 功耗≤5W |
| 论文撰写 | 14-15 | 毕业论文+答辩材料 | 论文格式规范，内容完整 |

## 技术难点与解决方案

| 难点 | 解决方案 |
|-----|---------|
| 量化后精度下降 | 混合精度量化 + 量化感知训练(QAT) |
| 实时性不达标 | 降低分辨率(640→416) + RGA加速 + SIMD优化 |
| 小目标检测困难 | 多尺度特征融合 + 调整Anchor尺寸 |
| NPU算子不支持 | ONNX图优化 + 算子映射到支持的算子 |
| Transpose CPU回退 | 使用416×416分辨率（<16384元素限制） |

## 质量保证措施

1. **代码质量**
   - 单元测试（pytest，88-100%覆盖率）
   - 代码规范（black、pylint、flake8）
   - 类型检查（mypy）

2. **性能监控**
   - 逐模块性能分析
   - 端到端延迟跟踪
   - NPU利用率监控

3. **文档完善**
   - 代码注释
   - API文档
   - 用户指南
   - 部署手册
