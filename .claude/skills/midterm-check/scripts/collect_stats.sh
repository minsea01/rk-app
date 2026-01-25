#!/bin/bash
# 中期检查统计脚本

echo "=== 中期检查数据统计 ==="
echo "日期: $(date '+%Y-%m-%d')"
echo ""

echo "=== 代码统计 ==="
echo "Python 代码行数:"
find apps/ tools/ -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1
echo ""
echo "Shell 脚本行数:"
find scripts/ -name "*.sh" -exec wc -l {} + 2>/dev/null | tail -1

echo ""
echo "=== Git 提交统计 ==="
echo "总提交数: $(git log --oneline --since='2024-10-01' | wc -l)"
echo ""

echo "=== 测试统计 ==="
echo "测试文件数: $(find tests/ -name 'test_*.py' | wc -l)"
echo "测试用例数: $(grep -r 'def test_' tests/ | wc -l)"

echo ""
echo "=== 文档统计 ==="
echo "Markdown 文档数: $(find docs/ -name '*.md' | wc -l)"
echo "论文章节数: $(find docs/thesis/ -name 'thesis_chapter_*.md' | wc -l)"

echo ""
echo "=== 模型文件 ==="
ls -lh artifacts/models/*.onnx artifacts/models/*.rknn 2>/dev/null || echo "无模型文件"
