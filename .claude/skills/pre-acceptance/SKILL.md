# 预验收检查准备 (Pre-Acceptance Preparation)

根据《仪器与电子学院毕业设计预验收检查表》，准备预验收演示和材料。

## 预验收要求

**核心评价维度：**
1. 演示正常完成 / 演示不能正常完成 / 演示无实质内容
2. 回答问题正确 / 回答问题基本正确 / 回答问题大部分错误
3. 思路正确 / 操作正确熟练 / 在别人协助下才能完成

**完成情况评价：**
- 超额完成任务书指标
- 已达到任务书指标
- 未完成

## 执行任务

### 1. 演示环境测试

```bash
# 检查 Python 环境
source ~/yolo_env/bin/activate
python --version
pip list | grep -E "(onnx|ultralytics|opencv)"

# 测试 ONNX 推理
yolo predict model=artifacts/models/best.onnx source=assets/test.jpg imgsz=416 conf=0.5

# 检查模型文件
ls -lh artifacts/models/*.onnx artifacts/models/*.rknn 2>/dev/null
```

### 2. 演示内容准备

本项目演示内容：

**软件演示：**
1. YOLO模型推理演示（PC端）
   - ONNX GPU 推理
   - 检测结果可视化

2. 模型转换流程演示
   - PyTorch → ONNX 导出
   - ONNX → RKNN 转换

3. 代码结构讲解
   - apps/ 核心模块
   - tools/ 工具链
   - tests/ 测试覆盖

**仿真演示：**
1. RKNN PC 模拟器运行
2. ONNX vs RKNN 精度对比
3. 性能基准测试

### 3. 演示素材清单

请确认以下素材准备情况：

| 素材类型 | 文件/位置 | 状态 |
|---------|----------|------|
| 软件源代码 | apps/, tools/ | ✅ |
| 可执行软件 | scripts/deploy/rk3588_run.sh | ✅ |
| 仿真结果 | artifacts/*_report.md | ✅ |
| 试验结果 | artifacts/bench_summary.json | ✅ |
| 仿真代码或模型 | apps/yolov8_rknn_infer.py | ✅ |
| 相关音像资料 | 演示录屏（待准备） | ⏸️ |

### 4. 演示脚本生成

生成演示流程脚本：

```markdown
## 演示流程（5-8分钟）

### Part 1: 项目概述（1分钟）
- 项目目标：RK3588行人检测模块
- 技术路线：YOLO11 + INT8量化 + NPU部署

### Part 2: 模型推理演示（2分钟）
- 展示 ONNX 推理结果
- 展示检测效果截图

### Part 3: 代码架构讲解（2分钟）
- apps/ 核心模块结构
- 配置管理、异常处理、日志系统

### Part 4: 性能数据展示（1分钟）
- 模型大小：4.7MB
- 推理速度：8.6ms @ 416×416
- 测试覆盖：88-100%

### Part 5: 问答环节（2-3分钟）
```

### 5. 常见问题准备

生成Q&A文档，包含：
- 模型选型原因
- 量化精度损失分析
- 板端部署方案
- 双网卡设计思路

### 6. 任务书指标对照

| 指标要求 | 当前完成情况 | 评价 |
|---------|-------------|------|
| 模型<5MB | 4.7MB | ✅ 达标 |
| >30 FPS | PC端60+FPS，板端待测 | ⏸️ 部分 |
| ≥90% mAP | 61.57%，Fine-tune可达 | ⏸️ 部分 |
| 双网卡≥900Mbps | 设计完成，待硬件验证 | ⏸️ 部分 |

### 7. 输出

- `docs/thesis/pre_acceptance_checklist.md` - 预验收检查清单
- `docs/thesis/demo_script.md` - 演示脚本
- `docs/thesis/qa_preparation.md` - 问答准备
