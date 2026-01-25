#!/bin/bash
# 佐证材料收集脚本

OUTPUT_DIR="docs/evidence"
mkdir -p "$OUTPUT_DIR"

echo "=== 佐证材料收集 ==="
echo "输出目录: $OUTPUT_DIR"
echo ""

# 代码统计
echo "收集代码统计..."
{
    echo "# 代码统计报告"
    echo "生成时间: $(date '+%Y-%m-%d %H:%M')"
    echo ""
    echo "## Python 模块"
    ls -la apps/*.py apps/utils/*.py 2>/dev/null
    echo ""
    echo "## 工具脚本"
    ls -la tools/*.py 2>/dev/null
    echo ""
    echo "## 部署脚本"
    find scripts/ -name "*.sh" | head -20
} > "$OUTPUT_DIR/code_stats.txt"

# Git 历史
echo "收集 Git 提交历史..."
git log --oneline --since="2024-10-01" > "$OUTPUT_DIR/git_history.txt"

# 测试报告
echo "运行测试..."
source ~/yolo_env/bin/activate 2>/dev/null
pytest tests/unit -v --tb=short > "$OUTPUT_DIR/test_report.txt" 2>&1 || true

echo ""
echo "=== 收集完成 ==="
echo "文件列表:"
ls -la "$OUTPUT_DIR/"
