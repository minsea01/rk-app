# Claude Code Skills for RK3588 Project

This directory contains specialized skills for automating common workflows in the RK3588 pedestrian detection graduation project.

## Available Skills

### 1. 🔄 full-pipeline
**Complete model conversion and validation pipeline**

Executes the full workflow from PyTorch/ONNX to RKNN with validation.

**When to use:**
- Converting a new model for deployment
- Running complete validation before board testing
- Generating baseline metrics for thesis

**What it does:**
- Export PyTorch → ONNX (if needed)
- Convert ONNX → RKNN with INT8 quantization
- Run PC simulator validation
- Generate comprehensive report

**Usage in Claude Code:**
```
Can you run the full-pipeline skill to convert my model?
```

---

### 2. ✅ board-ready
**Board deployment readiness check**

Comprehensive checklist to verify all components are ready for RK3588 deployment.

**When to use:**
- Before attempting board deployment
- After building ARM64 binaries
- To verify thesis technical requirements compliance

**What it checks:**
- ARM64 binary existence and correctness
- RKNN model files (<5MB requirement)
- Configuration files validity
- Deployment scripts executability
- Required libraries and dependencies

**Usage in Claude Code:**
```
Run the board-ready skill to check if I can deploy to the board
```

---

### 3. 📊 thesis-report
**Graduation thesis progress report generator**

Generates comprehensive progress reports aligned with graduation requirements.

**When to use:**
- Preparing midterm progress reports (中期检查)
- Updating advisor on project status
- Writing thesis chapters
- Before defense preparation

**What it generates:**
- Completion percentage by phase
- Technical requirements compliance table
- Performance metrics summary
- Risk assessment
- Timeline analysis

**Parameters:**
- `report_type`: "progress", "midterm1", "midterm2", "final"
- `include_code_stats`: Show code statistics

**Usage in Claude Code:**
```
Generate a thesis progress report using the thesis-report skill
```

---

### 4. ⚡ performance-test
**Comprehensive performance benchmark suite**

Runs all performance tests and generates consolidated metrics.

**When to use:**
- Validating model performance for thesis
- Comparing different configurations
- Preparing defense materials
- Optimizing hyperparameters

**What it tests:**
- ONNX GPU inference (baseline)
- RKNN PC simulator
- MCP benchmark pipeline
- Parameter tuning (conf/iou sweeps)

**Usage in Claude Code:**
```
Run the performance-test skill to benchmark all models
```

---

### 5. 🎯 model-validate
**Model accuracy validation and ONNX vs RKNN comparison**

Validates conversion quality and computes accuracy metrics.

**When to use:**
- Verifying INT8 quantization accuracy
- Checking ONNX→RKNN conversion correctness
- Computing mAP for thesis (requires ground truth)
- Creating visual results for defense

**What it validates:**
- Numerical output differences (MAE, max error)
- Detection-level agreement
- Visual comparison (side-by-side images)
- mAP@0.5 calculation (if ground truth provided)

**Usage in Claude Code:**
```
Validate my model using the model-validate skill
```

---

## 毕业设计专用 Skills (Graduation Design Skills)

以下 skills 根据《仪器与电子学院毕业设计工作手册》设计，每个 skill 包含完整的目录结构：

```
skill-name/
├── SKILL.md        ← 核心指令文件
├── scripts/        ← 可执行脚本
├── references/     ← 参考文档（工作手册原件）
└── assets/         ← 模板、图片等资源
```

### 6. 📋 midterm-check
**中期检查准备**

根据《阶段检查评分标准》准备中期检查材料。

**参考文档：**
- 附件6：仪器与电子学院毕业设计阶段中期检查表.docx
- 附件7：毕业设计阶段检查评分标准.docx
- 毕业设计03--中期总结格式(2024版).doc

**评分标准对照（第1阶段）：**
| 项目 | 分值 | 要求 |
|-----|------|-----|
| 辅导次数记录 | 10分 | ≥10次 |
| 记录本笔记页数 | 10分 | ≥10页 |
| 进度评价 | 50分 | 对照任务书 |
| 前期文档规范 | 30分 | 任务书+开题+中期 |

**Usage:**
```
/midterm-check
```

---

### 7. ✅ pre-acceptance
**预验收准备**

准备预验收演示和材料。

**参考文档：**
- 附件8：仪器与电子学院毕业设计预验收检查表.docx
- 附件9：毕业设计评价标准说明.docx

**核心评价维度：**
- 演示正常完成 / 演示不能正常完成
- 回答问题正确 / 基本正确 / 大部分错误
- 完成情况：超额完成 / 已达标 / 未完成

**Usage:**
```
/pre-acceptance
```

---

### 8. 🎓 defense-prep
**答辩准备**

生成PPT大纲、演讲稿和Q&A准备。

**参考文档：**
- 毕业设计06--说明书排版模板.doc
- 毕业设计07--撰写格式和内容的有关要求.doc
- 毕业设计08--本科毕业设计评价表.doc
- 附件15：毕业设计说明书抽检专家评议表.doc

**生成内容：**
- PPT大纲（20-25页，12-15分钟）
- 演讲稿（逐页要点）
- 常见问题Q&A

**Usage:**
```
/defense-prep
```

---

### 9. 📦 evidence-collect
**佐证材料收集**

自动收集和整理项目佐证材料。

**参考文档：**
- 附件19：学生毕业设计归档材料确认意见表.doc
- 附件16：仪器与电子学院毕业设计说明书形式审查表.doc

**收集内容：**
- 代码行数统计
- 测试覆盖率报告
- Git 提交历史
- 仿真结果整理

**Usage:**
```
/evidence-collect
```

---

### 10. 📝 progress-log
**进度日志生成**

根据 Git 提交生成周工作日志。

**用途：**
满足中期检查对"记录本笔记页数"的要求（第1阶段≥10页，第2阶段≥8页）

**生成内容：**
- 每周工作记录模板
- 辅导记录模板
- Git 提交按周汇总

**Usage:**
```
/progress-log
```

---

## How to Use Skills

### In Claude Code Conversations

Simply mention the skill in your request:

```
"Run the full-pipeline skill on my yolo11n model"
"Use the board-ready skill to check deployment status"
"Generate a thesis report with the thesis-report skill"
```

Claude will automatically:
1. Load the skill definition
2. Execute the workflow steps
3. Generate outputs and reports
4. Save artifacts to appropriate locations

### Typical Workflow

**Phase 1: Model Development**
```
1. Train model or download pretrained
2. "Run full-pipeline skill" → Convert to RKNN
3. "Run model-validate skill" → Check accuracy
4. "Run performance-test skill" → Benchmark
```

**Phase 2: Board Preparation**
```
1. Build ARM64 binary
2. "Run board-ready skill" → Verify all components
3. Deploy to board
4. Test on hardware
```

**Phase 3: Thesis Writing**
```
1. "Run thesis-report skill for midterm1" → Progress report
2. Continue development
3. "Run thesis-report skill for midterm2" → Second report
4. "Run thesis-report skill for final" → Final documentation
```

**Phase 4: Defense Preparation**
```
1. "Run performance-test skill" → Latest metrics
2. "Run model-validate skill with visualize" → Visual results
3. "Run thesis-report skill" → Complete summary
```

## Output Locations

Skills save their outputs to consistent locations:

```
artifacts/
├── models/                          # Models from full-pipeline
├── pipeline_report.md               # full-pipeline output
├── board_ready_report.md            # board-ready output
├── performance_report_{date}.md     # performance-test output
├── performance_metrics.json
├── validation_report_{date}.md      # model-validate output
├── onnx_vs_rknn_comparison.json
└── visualizations/                  # Visual comparisons

docs/
└── thesis_progress_report_{date}.md # thesis-report output
```

## Tips

1. **Run skills in sequence** for comprehensive validation:
   ```
   full-pipeline → model-validate → performance-test → board-ready
   ```

2. **Use thesis-report regularly** to track progress and prepare documentation

3. **Combine skills** for complex workflows:
   ```
   "After running full-pipeline, also run model-validate and generate a thesis report"
   ```

4. **Skills are idempotent**: Safe to run multiple times, they check existing outputs

5. **Check artifacts/** after running skills to find generated reports

## Customization

Skills are markdown files and can be easily customized:

1. Edit the skill file: `.claude/skills/{skill-name}.md`
2. Modify steps, parameters, or output formats
3. Skills automatically reload on next use

## Requirements

- Active `yolo_env` Python environment
- ONNX/RKNN models in `artifacts/models/`
- Test images in `assets/` or specified paths
- For board-ready: Built ARM64 binaries

## Troubleshooting

**Skill not found:**
- Ensure file exists in `.claude/skills/`
- Check skill name spelling

**Execution errors:**
- Check CLAUDE.md for environment setup
- Verify Python environment activated
- Check model files exist

**Output not generated:**
- Check `artifacts/` directory permissions
- Verify disk space available

## Support

For questions or issues with skills:
1. Check CLAUDE.md for project setup
2. Review skill markdown file for requirements
3. Check recent execution logs
4. Consult with Claude Code

---

Created for: RK3588 Pedestrian Detection Graduation Project
Institution: North University of China (中北大学)
Date: October 2025
