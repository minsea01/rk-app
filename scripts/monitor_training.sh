#!/bin/bash
# 训练进度监控工具

LOG_FILE=$(ls -t logs/auto_train_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ 未找到训练日志文件"
    exit 1
fi

echo "=========================================="
echo "训练进度监控"
echo "=========================================="
echo "日志文件: $LOG_FILE"
echo ""

# 检查训练是否在运行
if pgrep -f "auto_train_pedestrian.sh" > /dev/null; then
    PID=$(pgrep -f "auto_train_pedestrian.sh")
    echo "✓ 训练正在运行中 (PID: $PID)"
else
    echo "⚠ 训练未运行"
fi

echo ""
echo "=========================================="
echo "最新进度"
echo "=========================================="

# 显示最近的进度信息
tail -50 "$LOG_FILE" | grep -E "\[|✓|mAP|Epoch|下载" || echo "正在初始化..."

echo ""
echo "=========================================="
echo "监控选项"
echo "=========================================="
echo "  1) 实时查看日志: tail -f $LOG_FILE"
echo "  2) 查看完整日志: less $LOG_FILE"
echo "  3) 查看GPU使用: watch -n 1 nvidia-smi"
echo ""
