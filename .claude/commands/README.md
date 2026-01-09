# Claude Code Slash Commands

Custom slash commands for RK3588 project automation.

## Available Commands

### `/full-pipeline`
Execute complete model conversion pipeline (PyTorch → ONNX → RKNN → Validation)

**Usage:**
```
/full-pipeline
```

**What it does:**
- Checks prerequisites
- Exports/converts models
- Runs PC simulator validation
- Generates performance report

---

### `/board-ready`
Check RK3588 board deployment readiness

**Usage:**
```
/board-ready
```

**What it does:**
- Checks ARM64 binary
- Verifies RKNN models (<5MB)
- Validates configuration
- Checks deployment scripts
- Generates readiness report

---

### `/thesis-report`
Generate graduation thesis progress report

**Usage:**
```
/thesis-report
```

**What it does:**
- Analyzes project completion status
- Calculates progress percentage
- Compares vs graduation requirements
- Provides risk assessment
- Generates detailed report

---

### `/performance-test`
Run comprehensive performance benchmarks

**Usage:**
```
/performance-test
```

**What it does:**
- ONNX GPU inference test
- RKNN simulator test
- MCP benchmark pipeline
- Parameter tuning analysis
- Consolidated performance report

---

### `/model-validate`
Validate model accuracy (ONNX vs RKNN comparison)

**Usage:**
```
/model-validate
```

**What it does:**
- ONNX & RKNN inference comparison
- Numerical output analysis
- Visual validation (side-by-side)
- mAP calculation (if ground truth available)
- Accuracy validation report

---

## 毕业设计专用命令 (Graduation Design Commands)

以下命令根据《仪器与电子学院毕业设计工作手册》设计：

### `/midterm-check`
中期检查准备

**Usage:**
```
/midterm-check
```

**What it does:**
- 检查佐证材料完成情况
- 统计代码/文档/测试数据
- 对照评分标准评估进度
- 生成中期检查报告模板

---

### `/pre-acceptance`
预验收准备

**Usage:**
```
/pre-acceptance
```

**What it does:**
- 测试演示环境
- 准备演示内容和素材
- 生成演示脚本
- 对照任务书指标

---

### `/defense-prep`
答辩准备

**Usage:**
```
/defense-prep
```

**What it does:**
- 生成PPT大纲（20-25页）
- 生成演讲稿
- 准备常见问题Q&A
- 收集关键数据用于图表

---

### `/evidence-collect`
佐证材料收集

**Usage:**
```
/evidence-collect
```

**What it does:**
- 代码行数统计
- 测试覆盖率报告
- 仿真结果整理
- Git提交记录收集
- 生成材料清单

---

### `/progress-log`
进度日志生成

**Usage:**
```
/progress-log
```

**What it does:**
- 从Git提交生成周工作记录
- 适配毕业设计记录本格式
- 生成辅导记录模板
- 满足中期检查页数要求

---

## How to Use

Simply type the slash command in your Claude Code conversation:

```
User: /board-ready
Claude: [Executes deployment readiness check...]
```

Or describe what you want:

```
User: "Run the board readiness check"
Claude: [Recognizes intent and executes /board-ready...]
```

## Output Locations

Commands save outputs to standard locations:

```
artifacts/
├── board_ready_report.md         # /board-ready output
├── pipeline_report.md             # /full-pipeline output
├── performance_report_*.md        # /performance-test output
├── validation_report_*.md         # /model-validate output
└── performance_metrics.json

docs/
└── thesis_progress_report_*.md    # /thesis-report output
```

## Integration with Skills

These commands execute the workflows defined in `.claude/skills/`:
- Commands are the "slash command interface"
- Skills are the "detailed workflow definitions"
- Both work together for comprehensive automation

## Typical Workflow

**Model Development Phase:**
```
/full-pipeline       → Convert model
/model-validate      → Verify accuracy
/performance-test    → Benchmark performance
```

**Board Preparation:**
```
/board-ready         → Check deployment status
[Build ARM64 binary if needed]
[Deploy to board when hardware available]
```

**Thesis Writing:**
```
/thesis-report       → Generate progress report
[Write thesis chapters]
/thesis-report       → Update for midterm/final
```

**Defense Preparation:**
```
/performance-test    → Latest metrics
/model-validate      → Visual results
/thesis-report       → Complete summary
```

**毕业设计检查流程 (Graduation Design Check):**
```
/progress-log        → 生成工作日志（记录本用）
/evidence-collect    → 收集佐证材料
/midterm-check       → 中期检查准备（12月/4月）
/pre-acceptance      → 预验收准备（5月）
/defense-prep        → 答辩准备（6月）
```

## Troubleshooting

**Command not recognized:**
- Commands are in `.claude/commands/` directory
- Each command is a separate `.md` file
- Claude Code loads them automatically

**Execution errors:**
- Check CLAUDE.md for environment setup
- Verify Python environment: `source ~/yolo_env/bin/activate`
- Check model files exist in `artifacts/models/`

**No output generated:**
- Check `artifacts/` directory permissions
- Verify disk space
- Review command output for errors

## Customization

Commands are markdown files and can be customized:

1. Edit: `.claude/commands/{command-name}.md`
2. Modify task description or expected actions
3. Commands reload automatically on next use

## Support

- Check `CLAUDE.md` for project documentation
- Review `.claude/skills/README.md` for workflow details
- Consult command `.md` files for specific requirements

---

Created for: RK3588 Pedestrian Detection Graduation Project
Institution: North University of China (中北大学)
