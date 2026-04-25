#!/usr/bin/env bash
# Health checks should aggregate failures instead of exiting on the first one.
set -uo pipefail

# RK3588板上健康检查脚本
# 板子到手后第一个运行的脚本，5分钟内验证所有关键环节

ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "RK3588 Board Health Check"
echo "=========================================="
echo ""

PASS=0
FAIL=0

check() {
    local desc="$1"
    local cmd="$2"
    echo -n "[$desc] ... "
    if eval "$cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
        return 0
    else
        echo -e "${RED}FAIL${NC}"
        FAIL=$((FAIL + 1))
        return 1
    fi
}

check_or_continue() {
    check "$1" "$2" || true
}

first_model_path() {
    find "$ROOT/artifacts/models" -maxdepth 1 -type f -name '*.rknn' | sort | head -n 1
}

run_rknn_probe() {
    local model_path="$1"
    python3 - "$model_path" <<'PY'
import sys
from rknnlite.api import RKNNLite

model_path = sys.argv[1]
rknn = RKNNLite()
try:
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise SystemExit(ret)
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    if ret != 0:
        raise SystemExit(ret)
finally:
    try:
        rknn.release()
    except Exception:
        pass
PY
}

has_render_node() {
    compgen -G '/dev/dri/renderD*' > /dev/null
}

has_rknpu_device() {
    compgen -G '/dev/rknpu*' > /dev/null
}

# Layer 1: 基础环境
echo "=== Layer 1: 基础环境 ==="
check_or_continue "Python3安装" "command -v python3"
check_or_continue "Pip3安装" "command -v pip3"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python版本: $PYTHON_VERSION"

# Layer 2: Python依赖
echo ""
echo "=== Layer 2: Python依赖 ==="
check_or_continue "NumPy安装" "python3 -c 'import numpy'"
check_or_continue "OpenCV安装" "python3 -c 'import cv2'"
check_or_continue "Pillow安装" "python3 -c 'from PIL import Image'"
check_or_continue "YAML安装" "python3 -c 'import yaml'"

# Layer 3: RKNNLite
echo ""
echo "=== Layer 3: RKNN Runtime ==="
if check "RKNNLite导入" "python3 -c 'from rknnlite.api import RKNNLite'"; then
    MODEL_PATH="$(first_model_path)"
    if [[ -n "${MODEL_PATH:-}" ]]; then
        echo -n "[RKNN模型加载与NPU初始化] ... "
        if run_rknn_probe "$MODEL_PATH" > /dev/null 2>&1; then
            echo -e "${GREEN}PASS${NC}"
            PASS=$((PASS + 1))
            echo "  使用模型: ${MODEL_PATH#$ROOT/}"
        else
            echo -e "${RED}FAIL${NC}"
            FAIL=$((FAIL + 1))
            run_rknn_probe "$MODEL_PATH" 2>&1 | sed 's/^/  /'
        fi
    else
        echo "  未找到本地 .rknn 模型，跳过 NPU 初始化实测"
    fi
fi

# Layer 4: NPU硬件
echo ""
echo "=== Layer 4: NPU硬件 ==="
check_or_continue "NPU设备节点" "compgen -G '/dev/rknpu*' > /dev/null || compgen -G '/dev/dri/renderD*' > /dev/null"
if has_rknpu_device || has_render_node; then
    ls -l /dev/rknpu* /dev/dri/renderD* 2>/dev/null | sed 's/^/  /'
fi

check_or_continue "RKNPU驱动信息" "test -f /sys/kernel/debug/rknpu/version || dmesg 2>/dev/null | grep -qi rknpu"
if [ -f /sys/kernel/debug/rknpu/version ]; then
    sed 's/^/  /' /sys/kernel/debug/rknpu/version
elif dmesg 2>/dev/null | grep -qi rknpu; then
    dmesg 2>/dev/null | grep -i rknpu | tail -5 | sed 's/^/  /'
fi

# Layer 5: 平台检测
echo ""
echo "=== Layer 5: 平台信息 ==="
echo "  Architecture: $(uname -m)"
echo "  Kernel: $(uname -r)"
echo "  OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"

if [ -f /proc/device-tree/model ]; then
    echo "  Board Model: $(tr -d '\0' < /proc/device-tree/model)"
fi

# Layer 6: C++ 工具链与依赖
echo ""
echo "=== Layer 6: C++ 工具链与依赖 ==="
check_or_continue "gcc安装" "command -v gcc"
check_or_continue "g++安装" "command -v g++"
check_or_continue "cmake安装" "command -v cmake"
check_or_continue "make安装" "command -v make"
check_or_continue "pkg-config安装" "command -v pkg-config"
check_or_continue "OpenCV开发包" "pkg-config --exists opencv4 || pkg-config --exists opencv"
check_or_continue "yaml-cpp开发包" "pkg-config --exists yaml-cpp"
check_or_continue "GStreamer开发包" "pkg-config --exists gstreamer-1.0 gstreamer-app-1.0"
check_or_continue "RKNN运行库" "ldconfig -p 2>/dev/null | grep -q 'librknnrt\\.so' || find /usr /lib /opt -maxdepth 4 -name 'librknnrt.so*' | grep -q ."
check_or_continue "RKNN C SDK头文件" "test -f /opt/rknpu2/include/rknn_api.h || find /root /home /opt -maxdepth 6 -path '*/include/rknn_api.h' | grep -q ."

# Layer 7: 资源检查
echo ""
echo "=== Layer 7: 资源检查 ==="
MEM_TOTAL=$(free -h | awk '/^Mem:/ {print $2}')
MEM_AVAIL=$(free -h | awk '/^Mem:/ {print $7}')
echo "  内存总量: $MEM_TOTAL"
echo "  可用内存: $MEM_AVAIL"

DISK_AVAIL=$(df -h . | awk 'NR==2 {print $4}')
echo "  磁盘可用: $DISK_AVAIL"
check_or_continue "磁盘可用空间>=2G" "[ \$(df -Pk . | awk 'NR==2 {print \$4}') -ge 2097152 ]"

# Layer 8: 网络
echo ""
echo "=== Layer 8: 网络接口 ==="
ip link show | grep -E "^[0-9]+" | awk '{print $2}' | sed 's/:$//' | while read iface; do
    if [ "$iface" != "lo" ]; then
        STATE=$(ip link show "$iface" | grep -o "state [A-Z]*" | awk '{print $2}')
        echo "  $iface: $STATE"
    fi
done

# 汇总
echo ""
echo "=========================================="
echo "总计: ${GREEN}${PASS} PASS${NC}, ${RED}${FAIL} FAIL${NC}"
echo "=========================================="

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✅ 板子完全ready，可以开始部署${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  有${FAIL}项检查失败，需要先修复${NC}"
    echo ""
    echo "常见修复方法："
    echo "  - Python 依赖缺失: pip3 install -r requirements_board.txt"
    echo "  - 缺少 C++ 开发包: sudo apt-get install -y libyaml-cpp-dev libopencv-dev pkg-config"
    echo "  - 缺少 GStreamer 头文件: sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev"
    echo "  - 缺少 RKNN C SDK: 设置 RKNN_HOME 或安装完整 /opt/rknpu2 SDK（仅有 runtime 库不足以编译 C++）"
    echo "  - 注意: 新版 RKNPU 可能通过 /dev/dri/renderD* 暴露，不一定存在 /dev/rknpu0"
    exit 1
fi
