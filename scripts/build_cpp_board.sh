#!/bin/bash
# 板端原生编译C++项目（RK3588）
#
# 使用方法（在板端SSH执行）：
#   cd ~/rk-app
#   bash scripts/build_cpp_board.sh

set -e

echo "=================================================="
echo "RK3588板端原生编译"
echo "=================================================="

# 检查架构
ARCH=$(uname -m)
if [[ "$ARCH" != "aarch64" ]]; then
    echo "❌ 错误: 当前架构是 $ARCH, 必须在 aarch64 (ARM64) 板端运行"
    exit 1
fi

echo "✓ 架构检查: $ARCH"

# 检查必要工具
echo ""
echo "检查编译工具..."
USE_NINJA="OFF"
for tool in cmake g++; do
    if command -v $tool &> /dev/null; then
        echo "  ✓ $tool"
    else
        echo "  ❌ $tool 未安装"
        MISSING_TOOLS=1
    fi
done

if command -v ninja &> /dev/null; then
    echo "  ✓ ninja"
    USE_NINJA="ON"
else
    echo "  ⚠️  ninja 未安装，将使用Unix Makefiles"
    USE_NINJA="OFF"
fi

if [ ! -z "$MISSING_TOOLS" ]; then
    echo ""
    echo "请安装缺失的工具:"
    echo "  sudo apt-get install cmake g++"
    exit 1
fi

# 检查RKNN SDK
RKNN_HOME="/opt/rknpu2"
if [ ! -d "$RKNN_HOME" ]; then
    echo ""
    echo "⚠️  警告: RKNN SDK 未找到 ($RKNN_HOME)"
    echo "   项目将以RKNN禁用模式编译（仅ONNX）"
    echo ""
    ENABLE_RKNN="OFF"
else
    echo "  ✓ RKNN SDK: $RKNN_HOME"
    ENABLE_RKNN="ON"
fi

# 配置CMake
echo ""
echo "=================================================="
echo "配置CMake..."
echo "=================================================="

rm -rf build/board
mkdir -p build/board

if [ "$USE_NINJA" = "ON" ]; then
    CMAKE_GENERATOR="Ninja"
else
    CMAKE_GENERATOR="Unix Makefiles"
fi

echo "使用生成器: ${CMAKE_GENERATOR}"

cmake -B build/board \
    -G "${CMAKE_GENERATOR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_RKNN=${ENABLE_RKNN} \
    -DENABLE_ONNX=OFF \
    -DRKNN_HOME=${RKNN_HOME} \
    -DCMAKE_INSTALL_PREFIX=out/board

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ CMake配置失败"
    exit 1
fi

# 编译
echo ""
echo "=================================================="
echo "编译项目..."
echo "=================================================="

cmake --build build/board --parallel $(nproc)

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 编译失败"
    exit 1
fi

# 检查生成的二进制
echo ""
echo "=================================================="
echo "✅ 编译成功！"
echo "=================================================="
echo ""
echo "生成的可执行文件:"
ls -lh build/board/detect_cli 2>/dev/null || echo "  ⚠️  detect_cli 未生成"
ls -lh build/board/detect_rknn_multicore 2>/dev/null || echo "  ⚠️  detect_rknn_multicore 未生成"

echo ""
echo "下一步："
echo "  1. 测试推理性能:"
echo "     ./build/board/detect_cli --cfg config/detect.yaml --source assets/test.jpg"
echo ""
echo "  2. 或运行基准测试:"
echo "     ./build/board/detect_cli --cfg config/detect.yaml --source assets/test.jpg --warmup 10"
echo ""
echo "=================================================="
