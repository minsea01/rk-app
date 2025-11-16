#!/bin/bash
# 启动自动训练（可后台运行）

echo "=========================================="
echo "自动训练启动器"
echo "=========================================="
echo ""

# 检查是否已有训练在运行
if pgrep -f "auto_train_pedestrian.sh" > /dev/null; then
    echo "⚠ 检测到训练正在运行中！"
    echo ""
    echo "查看进程："
    ps aux | grep auto_train_pedestrian.sh | grep -v grep
    echo ""
    echo "查看日志："
    echo "  tail -f logs/auto_train_*.log"
    exit 1
fi

echo "启动模式选择："
echo "  1) 前台运行（查看实时输出）"
echo "  2) 后台运行（推荐，可关闭终端）"
echo ""
read -p "请选择 [1/2]: " choice

case $choice in
    1)
        echo ""
        echo "=========================================="
        echo "前台运行模式"
        echo "=========================================="
        echo "按 Ctrl+C 可中断训练"
        echo ""
        bash scripts/auto_train_pedestrian.sh
        ;;
    2)
        echo ""
        echo "=========================================="
        echo "后台运行模式"
        echo "=========================================="
        
        # 后台运行
        nohup bash scripts/auto_train_pedestrian.sh > /dev/null 2>&1 &
        PID=$!
        
        echo "✓ 训练已在后台启动！"
        echo ""
        echo "进程ID: $PID"
        echo "日志文件: logs/auto_train_$(date +%Y%m%d)*.log"
        echo ""
        echo "监控命令："
        echo "  查看最新日志: tail -f logs/auto_train_*.log"
        echo "  查看进程: ps aux | grep $PID"
        echo "  停止训练: kill $PID"
        echo ""
        echo "预计完成时间: 约6-12小时后"
        echo "=========================================="
        
        # 保存PID
        echo $PID > logs/training.pid
        ;;
    *)
        echo "无效选择！"
        exit 1
        ;;
esac
