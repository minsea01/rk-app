# 佐证材料收集 (Evidence Collection)

根据毕业设计检查要求，自动收集和整理项目佐证材料。

## 佐证材料类别

根据《中期检查表》，佐证材料包括：
- 毕业设计记录本
- 开题报告、中期总结报告
- 软件代码、仿真结果
- 试验方案、调研报告
- 仿真代码或模型

## 执行任务

### 1. 代码统计

```bash
# 代码总量
find apps/ tools/ scripts/ -name "*.py" -exec wc -l {} + | tail -1
find apps/ tools/ scripts/ -name "*.sh" -exec wc -l {} + | tail -1

# 模块清单
ls -la apps/*.py apps/utils/*.py
ls -la tools/*.py
```

### 2. 测试报告生成

```bash
source ~/yolo_env/bin/activate
pytest tests/unit -v --cov=apps --cov=tools --cov-report=html
```

### 3. 仿真结果收集

```bash
ls -la artifacts/*.md artifacts/*.json
yolo predict model=artifacts/models/best.onnx source=assets/test.jpg imgsz=416 conf=0.5
```

### 4. 文档材料清单

| 文档 | 文件位置 | 状态 |
|-----|---------|------|
| 开题报告 | docs/thesis/thesis_opening_report.md | ✅ |
| 中期总结1 | docs/thesis/midterm_report_1.md | ✅ |
| 中期总结2 | docs/thesis/midterm_report_2.md | ✅ |
| 论文章节 | docs/thesis/thesis_chapter_*.md | ✅ |

### 5. Git提交记录

```bash
git log --oneline --since="2024-10-01" | head -30
git shortlog -sn --since="2024-10-01"
```

### 6. 输出

- `docs/evidence/evidence_summary.md` - 佐证材料汇总
- `artifacts/evidence_collection.json` - 结构化数据
