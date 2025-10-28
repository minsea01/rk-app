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
