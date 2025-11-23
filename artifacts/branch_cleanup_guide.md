# 分支清理指南

**生成时间:** 2025-11-23

## 待删除的分支（11个）

由于环境权限限制，无法通过命令行删除远程分支。请选择以下方式之一：

---

## 方法 1: 使用脚本（需要本地权限）

在本地有权限的环境中运行：

```bash
bash scripts/delete_low_priority_branches.sh
```

该脚本会交互式确认并删除以下分支。

---

## 方法 2: GitHub 网页删除（推荐）

访问 GitHub 仓库页面：https://github.com/minsea01/rk-app/branches

### 需要删除的分支列表：

**中优先级分支（6个）：**
1. `claude/review-project-completion-017TgbDVPj7obFiafDMMZQy1`
2. `claude/claude-md-mi5zrdhlk5jvz1rl-012aDjJ9SYRjMmnGfJCJPBJe`
3. `claude/testing-mi42h0ldprzwfqd2-01YWENqgRW6tci1umNFBM5RR`
4. `claude/high-standard-code-review-01JoqBEBB9jbGUz8R26uZUTf`
5. `claude/testing-mi2uei38kd9sj24h-01Q5pkxstAjCRhzjNdxN2CEa`
6. `claude/rk3588-pedestrian-detection-01G19RdwC5ZerdRuXvKK5p4J`

**低优先级分支（4个）：**
7. `claude/rk3588-pedestrian-detection-015LmRNMoGUj8AA7GoGKRySb`
8. `claude/testing-mi1goracy55rk0b0-012bH1ZqTCx9gXTMw7gEfE6Q`
9. `claude/rk3588-pedestrian-detection-01KpGGhptnTxNA2MRrmzeYPN`
10. `claude/claude-md-mi42gordjeazcups-01WzDLW4HGutuSdzwA14FsfA`

**已过时分支（1个）：**
11. `codex/review-graduation-project-feasibility`

### GitHub 网页删除步骤：

1. 打开 https://github.com/minsea01/rk-app/branches
2. 找到上述分支名称
3. 点击分支右侧的 🗑️ 删除图标
4. 确认删除

---

## 方法 3: 使用 gh CLI（如果已安装）

```bash
# 批量删除
gh api repos/minsea01/rk-app/git/refs/heads/claude/claude-md-mi42gordjeazcups-01WzDLW4HGutuSdzwA14FsfA -X DELETE
gh api repos/minsea01/rk-app/git/refs/heads/claude/claude-md-mi5zrdhlk5jvz1rl-012aDjJ9SYRjMmnGfJCJPBJe -X DELETE
gh api repos/minsea01/rk-app/git/refs/heads/claude/high-standard-code-review-01JoqBEBB9jbGUz8R26uZUTf -X DELETE
gh api repos/minsea01/rk-app/git/refs/heads/claude/review-project-completion-017TgbDVPj7obFiafDMMZQy1 -X DELETE
gh api repos/minsea01/rk-app/git/refs/heads/claude/rk3588-pedestrian-detection-015LmRNMoGUj8AA7GoGKRySb -X DELETE
gh api repos/minsea01/rk-app/git/refs/heads/claude/rk3588-pedestrian-detection-01G19RdwC5ZerdRuXvKK5p4J -X DELETE
gh api repos/minsea01/rk-app/git/refs/heads/claude/rk3588-pedestrian-detection-01KpGGhptnTxNA2MRrmzeYPN -X DELETE
gh api repos/minsea01/rk-app/git/refs/heads/claude/testing-mi1goracy55rk0b0-012bH1ZqTCx9gXTMw7gEfE6Q -X DELETE
gh api repos/minsea01/rk-app/git/refs/heads/claude/testing-mi2uei38kd9sj24h-01Q5pkxstAjCRhzjNdxN2CEa -X DELETE
gh api repos/minsea01/rk-app/git/refs/heads/claude/testing-mi42h0ldprzwfqd2-01YWENqgRW6tci1umNFBM5RR -X DELETE
gh api repos/minsea01/rk-app/git/refs/heads/codex/review-graduation-project-feasibility -X DELETE
```

---

## 保留的高优先级分支（4个）

这些分支包含重要的最新更新，**不要删除**：

✅ `claude/add-claude-documentation-01KVi7xtTks4wCMhiZmiDFUx` (2025-11-22)
   - 最新 CLAUDE.md 更新

✅ `claude/code-review-standards-01Lk3keunNjzN9C1DViJN9Xd` (2025-11-21)
   - RKNN-Toolkit2 部署报告

✅ `claude/wsl-to-rk3588-deployment-01QYhu2AbY36HHEoJmgB1CdD` (2025-11-21)
   - RK3588 部署指南和脚本

✅ `claude/yolov8-eval-testing-017eb6B9vGoC7WwPuXaBzMwy` (2025-11-20)
   - 测试闭环改进

---

## 删除后的后续步骤

1. **合并高优先级分支**
   ```bash
   git checkout master
   git pull origin master
   git merge --no-ff origin/claude/add-claude-documentation-01KVi7xtTks4wCMhiZmiDFUx
   git merge --no-ff origin/claude/code-review-standards-01Lk3keunNjzN9C1DViJN9Xd
   git merge --no-ff origin/claude/wsl-to-rk3588-deployment-01QYhu2AbY36HHEoJmgB1CdD
   git merge --no-ff origin/claude/yolov8-eval-testing-017eb6B9vGoC7WwPuXaBzMwy
   git push origin master
   ```

2. **删除已合并的高优先级分支**（合并后）
   ```bash
   git push origin --delete claude/add-claude-documentation-01KVi7xtTks4wCMhiZmiDFUx
   git push origin --delete claude/code-review-standards-01Lk3keunNjzN9C1DViJN9Xd
   git push origin --delete claude/wsl-to-rk3588-deployment-01QYhu2AbY36HHEoJmgB1CdD
   git push origin --delete claude/yolov8-eval-testing-017eb6B9vGoC7WwPuXaBzMwy
   ```

3. **验证清理结果**
   ```bash
   git fetch --prune
   git branch -r --no-merged origin/master
   ```

---

**推荐方式:** 使用 GitHub 网页删除（最简单、最直观）
