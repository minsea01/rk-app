# rk-app (RK3588 无板模板)

## 开发流程
- 本机 x86 调试：`Ctrl+Shift+B → x86: run`，F5 断点。
- 交叉 arm64 验证：`Ctrl+Shift+B → arm64: run (qemu)`。
- 单测：`ctest --preset x86-debug`。

## 目录
- src/, include/, tests/, config/
- scripts/: 脚本总目录
  - run/: 运行脚本（x86/ARM64 本地/QEMU）
  - deploy/: 上板部署、同步 sysroot
  - tune/: 调参与绘图脚本
  - 顶层脚本保留兼容包装器（原路径仍可用）
- results/: 数据产出（调参 JSON/CSV/PNG 等）
- .vscode/: VS Code 任务与调试
- CMakePresets.json: x86 / arm64 双预设

## 上板（等拿到板子）
1) `cmake --build --preset arm64 && cmake --install build/arm64`
2) `scripts/deploy_to_board.sh --host <board_ip>` （或 `scripts/deploy/deploy_to_board.sh`）
3) 板子运行或 `gdbserver :1234 ...` + VS Code 远程 attach

## 自动调参与输出
- 快速出图（低超调）：`python3 scripts/tune/auto_tune_pid.py --profile lowovershoot --plot-best`
- 快速出图（近 0.11s 上升）：`python3 scripts/tune/auto_tune_pid.py --profile fast100 --plot-best`
- 产出文件位于 `results/autotune/`，同时拷贝一份到 `out/` 以便兼容原有查看路径。
