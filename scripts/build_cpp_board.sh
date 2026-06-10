#!/usr/bin/env bash
# 板端原生编译C++项目（RK3588）
#
# 使用方法（在板端SSH执行）：
#   cd ~/rk-app
#   bash scripts/build_cpp_board.sh

set -euo pipefail

ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT"

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
MISSING_TOOLS=0
for tool in cmake g++ pkg-config; do
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

if [[ "$MISSING_TOOLS" -ne 0 ]]; then
    echo ""
    echo "请安装缺失的工具:"
    echo "  sudo apt-get install cmake g++ pkg-config"
    exit 1
fi

find_rknn_sdk() {
    local candidate found
    for candidate in \
        "${RKNN_HOME:-}" \
        /opt/rknpu2 \
        /home/RKnpuProjects/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api \
        /root/rk3588_linux_aarch64 \
        "$HOME/rk3588_linux_aarch64"; do
        [[ -n "${candidate}" && -d "${candidate}" ]] || continue
        if [[ -f "${candidate}/include/rknn_api.h" && ( -d "${candidate}/lib" || -d "${candidate}/aarch64" ) ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
        found="$(find "${candidate}" -maxdepth 5 -type f -name rknn_api.h 2>/dev/null | head -n 1 || true)"
        if [[ -n "${found}" ]]; then
            found="$(cd "$(dirname "${found}")"/.. && pwd)"
            if [[ -d "${found}/lib" || -d "${found}/aarch64" ]]; then
                printf '%s\n' "${found}"
                return 0
            fi
        fi
    done
    return 1
}

# 检查RKNN SDK
RKNN_HOME_DETECTED="$(find_rknn_sdk || true)"
if [[ -z "${RKNN_HOME_DETECTED}" ]]; then
    echo ""
    echo "⚠️  警告: RKNN SDK 未找到（未发现 include/rknn_api.h + lib/）"
    echo "   项目将以RKNN禁用模式编译（仅ONNX）"
    echo ""
    ENABLE_RKNN="OFF"
else
    RKNN_HOME="${RKNN_HOME_DETECTED}"
    echo "  ✓ RKNN SDK: $RKNN_HOME"
    ENABLE_RKNN="ON"
fi

if pkg-config --exists gstreamer-1.0 gstreamer-app-1.0; then
    ENABLE_GIGE="${ENABLE_GIGE:-ON}"
    ENABLE_CSI="${ENABLE_CSI:-ON}"
    echo "  ✓ GStreamer开发包"
else
    ENABLE_GIGE="${ENABLE_GIGE:-OFF}"
    ENABLE_CSI="${ENABLE_CSI:-OFF}"
    echo "  ⚠️  GStreamer开发包未找到，GigE/CSI source 将被禁用"
fi

if pkg-config --exists yaml-cpp; then
    echo "  ✓ yaml-cpp开发包"
else
    echo "  ❌ yaml-cpp 开发包未安装（需要 libyaml-cpp-dev）"
    exit 1
fi

if pkg-config --exists opencv4 || pkg-config --exists opencv; then
    echo "  ✓ OpenCV开发包"
else
    echo "  ❌ OpenCV 开发包未安装（需要 libopencv-dev）"
    exit 1
fi

# 配置CMake
echo ""
echo "=================================================="
echo "配置CMake..."
echo "=================================================="

rm -rf build/board
mkdir -p build/board

CMAKE_EXTRA_FLAGS=()
if [[ -d /usr/include/rga ]]; then
    CMAKE_EXTRA_FLAGS+=("-DCMAKE_CXX_FLAGS=-I/usr/include/rga")
fi

if [ "$USE_NINJA" = "ON" ]; then
    CMAKE_GENERATOR="Ninja"
else
    CMAKE_GENERATOR="Unix Makefiles"
fi

echo "使用生成器: ${CMAKE_GENERATOR}"
echo "ENABLE_RKNN=${ENABLE_RKNN} ENABLE_GIGE=${ENABLE_GIGE} ENABLE_CSI=${ENABLE_CSI}"

cmake -B build/board \
    -G "${CMAKE_GENERATOR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DENABLE_RKNN=${ENABLE_RKNN} \
    -DENABLE_ONNX=OFF \
    -DENABLE_GIGE=${ENABLE_GIGE} \
    -DENABLE_CSI=${ENABLE_CSI} \
    -DRKNN_HOME=${RKNN_HOME} \
    -DCMAKE_INSTALL_PREFIX=out/board \
    "${CMAKE_EXTRA_FLAGS[@]}"

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
echo "  1. 无相机 smoke 测试:"
echo "     ./build/board/detect_cli --cfg config/detection/detect_fake_camera.yaml --warmup 0"
echo ""
echo "  2. 真实 GigE 双网口链路:"
echo "     ./build/board/detect_cli --cfg config/detection/detect_rknn.yaml"
echo ""
echo "=================================================="
