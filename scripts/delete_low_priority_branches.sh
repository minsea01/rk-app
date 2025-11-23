#!/bin/bash
# 删除低优先级未合并分支脚本
# 生成时间: 2025-11-23
# 保留高优先级分支，删除其余11个分支

set -e

echo "=== 准备删除低优先级分支 ==="
echo ""
echo "保留的高优先级分支："
echo "  ✓ claude/add-claude-documentation-01KVi7xtTks4wCMhiZmiDFUx"
echo "  ✓ claude/code-review-standards-01Lk3keunNjzN9C1DViJN9Xd"
echo "  ✓ claude/wsl-to-rk3588-deployment-01QYhu2AbY36HHEoJmgB1CdD"
echo "  ✓ claude/yolov8-eval-testing-017eb6B9vGoC7WwPuXaBzMwy"
echo ""

# 要删除的分支列表
branches_to_delete=(
    "claude/claude-md-mi42gordjeazcups-01WzDLW4HGutuSdzwA14FsfA"
    "claude/claude-md-mi5zrdhlk5jvz1rl-012aDjJ9SYRjMmnGfJCJPBJe"
    "claude/high-standard-code-review-01JoqBEBB9jbGUz8R26uZUTf"
    "claude/review-project-completion-017TgbDVPj7obFiafDMMZQy1"
    "claude/rk3588-pedestrian-detection-015LmRNMoGUj8AA7GoGKRySb"
    "claude/rk3588-pedestrian-detection-01G19RdwC5ZerdRuXvKK5p4J"
    "claude/rk3588-pedestrian-detection-01KpGGhptnTxNA2MRrmzeYPN"
    "claude/testing-mi1goracy55rk0b0-012bH1ZqTCx9gXTMw7gEfE6Q"
    "claude/testing-mi2uei38kd9sj24h-01Q5pkxstAjCRhzjNdxN2CEa"
    "claude/testing-mi42h0ldprzwfqd2-01YWENqgRW6tci1umNFBM5RR"
    "codex/review-graduation-project-feasibility"
)

echo "将要删除的分支 (${#branches_to_delete[@]} 个):"
for branch in "${branches_to_delete[@]}"; do
    echo "  ✗ $branch"
done
echo ""

read -p "确认删除这些分支？ (y/N): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "操作已取消"
    exit 0
fi

echo ""
echo "开始删除分支..."
echo ""

deleted_count=0
failed_count=0

for branch in "${branches_to_delete[@]}"; do
    echo -n "删除 $branch ... "
    if git push origin --delete "$branch" 2>/dev/null; then
        echo "✓ 成功"
        ((deleted_count++))
    else
        echo "✗ 失败"
        ((failed_count++))
    fi
done

echo ""
echo "=== 删除完成 ==="
echo "成功: $deleted_count"
echo "失败: $failed_count"
echo ""

if [ $failed_count -eq 0 ]; then
    echo "所有低优先级分支已删除完成！"
else
    echo "部分分支删除失败，可能需要在 GitHub 网页上手动删除。"
fi
