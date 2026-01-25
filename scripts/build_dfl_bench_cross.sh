#!/bin/bash
# 交叉编译DFL优化性能测试 (WSL -> RK3588)
#
# 使用方法（在WSL执行）：
#   cd ~/rk-app
#   bash scripts/build_dfl_bench_cross.sh

set -e

echo "=================================================="
echo "交叉编译DFL优化性能测试 (x86_64 -> ARM64)"
echo "=================================================="

# 检查交叉编译器
if ! command -v aarch64-linux-gnu-g++ &> /dev/null; then
    echo "❌ 未找到交叉编译器 aarch64-linux-gnu-g++"
    echo ""
    echo "请安装:"
    echo "  sudo apt-get install g++-aarch64-linux-gnu"
    exit 1
fi

echo "✓ 交叉编译器检查通过"

# 编译
echo ""
echo "编译中..."
aarch64-linux-gnu-g++ -std=c++17 -O3 \
    -march=armv8.2-a+crypto+fp16 \
    -mtune=cortex-a76 \
    -ffast-math \
    -ftree-vectorize \
    -I include \
    examples/bench_dfl_optimized.cpp \
    src/infer/rknn/RknnDecodeOptimized.cpp \
    -o bench_dfl_opt \
    -static \
    -lpthread

if [ $? -eq 0 ]; then
    # 验证是ARM64二进制
    FILE_INFO=$(file bench_dfl_opt)
    if [[ "$FILE_INFO" == *"ARM aarch64"* ]]; then
        echo ""
        echo "=================================================="
        echo "✅ 交叉编译成功！"
        echo "=================================================="
        echo ""
        echo "二进制文件: bench_dfl_opt"
        echo "架构: ARM64 (aarch64)"
        echo "大小: $(ls -lh bench_dfl_opt | awk '{print $5}')"
        echo ""
        echo "下一步："
        echo "  1. 传输到板端:"
        echo "     scp bench_dfl_opt root@192.168.137.226:~/rk-app/"
        echo ""
        echo "  2. 板端运行:"
        echo "     ssh root@192.168.137.226"
        echo "     cd ~/rk-app"
        echo "     ./bench_dfl_opt"
        echo "=================================================="
    else
        echo "⚠️  警告: 编译产物不是ARM64架构"
        echo "$FILE_INFO"
        exit 1
    fi
else
    echo "❌ 编译失败"
    exit 1
fi
