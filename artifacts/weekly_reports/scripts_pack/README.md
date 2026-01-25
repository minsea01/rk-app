# RK3588 毕业设计脚本包

## 文件列表

| 文件 | 说明 |
|------|------|
| `demo_npu.sh` | 板端NPU演示脚本，直接在板子上运行 |
| `midterm_onestep.sh` | 中期检查一键脚本 |
| `midterm_report_20260107.md` | 中期检查报告 |
| `yolov8_rknn_infer.py` | NPU推理脚本（已修复batch维度） |
| `transfer_to_board.py` | 快速传输文件到板子 |
| `run_on_board.py` | 远程执行板子命令 |

## 使用方法

### 1. 板端演示
```bash
# SSH登录板子后
bash /root/demo_npu.sh
```

### 2. 远程执行
```bash
# 在WSL/PC上
python3 run_on_board.py "bash /root/demo_npu.sh"
```

### 3. 传输文件
```bash
python3 transfer_to_board.py local_file.rknn /root/rk-app/artifacts/models/
```

## 板子信息
- IP: 192.168.137.226
- 用户: root
- 密码: 123456

## 关键指标
- 推理延时: 25.31ms (≤45ms ✅)
- 帧率: 39.5 FPS (>30 FPS ✅)
- 模型大小: 4.3MB (<5MB ✅)
