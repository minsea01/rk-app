#!/bin/bash
# 生成周工作日志脚本

OUTPUT_DIR="docs/thesis/progress_log"
mkdir -p "$OUTPUT_DIR"

echo "=== 周工作日志生成 ==="

# 获取最近的提交按周分组
git log --oneline --since="2024-10-01" --format="%ad | %s" --date=format:'%Y-W%W' | \
while read line; do
    week=$(echo "$line" | cut -d'|' -f1 | tr -d ' ')
    msg=$(echo "$line" | cut -d'|' -f2-)
    echo "$week: $msg"
done | sort | uniq > "$OUTPUT_DIR/commits_by_week.txt"

# 生成周报告模板
cat > "$OUTPUT_DIR/weekly_template.md" << 'EOF'
# 第X周工作日志（MM.DD - MM.DD）

## 本周工作内容
1.
2.
3.

## 完成情况
- [x]
- [ ]

## 遇到问题及解决
- **问题**：
- **解决**：

## 下周计划
1.
2.

---
**指导教师意见**：

签名：__________ 日期：__________
EOF

echo "模板已生成: $OUTPUT_DIR/weekly_template.md"
echo "提交记录: $OUTPUT_DIR/commits_by_week.txt"
