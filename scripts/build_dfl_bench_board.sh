#!/bin/bash
# 板端快速编译DFL优化性能测试
#
# 使用方法（在板端执行）：
#   cd ~/rk-app
#   bash scripts/build_dfl_bench_board.sh
#   ./bench_dfl_opt

set -e

echo "=================================================="
echo "编译DFL优化性能测试 (RK3588板端)"
echo "=================================================="

# 检查架构
ARCH=$(uname -m)
if [[ "$ARCH" != "aarch64" ]]; then
    echo "⚠️  警告: 当前架构是 $ARCH, 需要在 aarch64 (ARM64) 板端运行"
    echo "   如果在PC上，请使用交叉编译"
    exit 1
fi

echo "✓ 架构检查通过: $ARCH"

# 编译
echo ""
echo "编译中..."
g++ -std=c++17 -O3 -march=armv8-a+fp+simd \
    -I include \
    examples/bench_dfl_optimized.cpp \
    src/infer/rknn/RknnDecodeOptimized.cpp \
    -o bench_dfl_opt \
    -lpthread

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ 编译成功！"
    echo "=================================================="
    echo ""
    echo "运行测试:"
    echo "  ./bench_dfl_opt"
    echo ""
    echo "预期结果:"
    echo "  - 标量版本: ~30ms/frame"
    echo "  - NEON优化: ~8-10ms/frame"
    echo "  - 加速比: 3-4x"
    echo "=================================================="
else
    echo "❌ 编译失败"
    exit 1
fi
