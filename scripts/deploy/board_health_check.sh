#!/usr/bin/env bash
set -euo pipefail

# RK3588板上健康检查脚本
# 板子到手后第一个运行的脚本，5分钟内验证所有关键环节

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
        ((PASS++))
        return 0
    else
        echo -e "${RED}FAIL${NC}"
        ((FAIL++))
        return 1
    fi
}

# Layer 1: 基础环境
echo "=== Layer 1: 基础环境 ==="
check "Python3安装" "command -v python3"
check "Pip3安装" "command -v pip3"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python版本: $PYTHON_VERSION"

# Layer 2: Python依赖
echo ""
echo "=== Layer 2: Python依赖 ==="
check "NumPy安装" "python3 -c 'import numpy'"
check "OpenCV安装" "python3 -c 'import cv2'"
check "Pillow安装" "python3 -c 'from PIL import Image'"
check "YAML安装" "python3 -c 'import yaml'"

# Layer 3: RKNNLite
echo ""
echo "=== Layer 3: RKNN Runtime ==="
if check "RKNNLite导入" "python3 -c 'from rknnlite.api import RKNNLite'"; then
    # 尝试创建实例
    python3 -c "
from rknnlite.api import RKNNLite
rknn = RKNNLite()
print('  RKNNLite实例创建: OK')
ret = rknn.init_runtime()
if ret == 0:
    print('  NPU初始化: OK')
else:
    print('  NPU初始化: FAILED (ret={})'.format(ret))
rknn.release()
" 2>&1 | grep -v "^$" | sed 's/^/  /'
fi

# Layer 4: NPU硬件
echo ""
echo "=== Layer 4: NPU硬件 ==="
check "NPU设备文件" "test -e /dev/rknpu0"
if [ -e /dev/rknpu0 ]; then
    ls -l /dev/rknpu* | sed 's/^/  /'
fi

check "RKNPU驱动模块" "lsmod | grep -q rknpu"
if lsmod | grep -q rknpu; then
    lsmod | grep rknpu | sed 's/^/  /'
fi

# Layer 5: 平台检测
echo ""
echo "=== Layer 5: 平台信息 ==="
echo "  Architecture: $(uname -m)"
echo "  Kernel: $(uname -r)"
echo "  OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"

if [ -f /proc/device-tree/model ]; then
    echo "  Board Model: $(cat /proc/device-tree/model)"
fi

# Layer 6: 内存检查
echo ""
echo "=== Layer 6: 资源检查 ==="
MEM_TOTAL=$(free -h | awk '/^Mem:/ {print $2}')
MEM_AVAIL=$(free -h | awk '/^Mem:/ {print $7}')
echo "  内存总量: $MEM_TOTAL"
echo "  可用内存: $MEM_AVAIL"

DISK_AVAIL=$(df -h . | awk 'NR==2 {print $4}')
echo "  磁盘可用: $DISK_AVAIL"

# Layer 7: 网络
echo ""
echo "=== Layer 7: 网络接口 ==="
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
    echo "  - RKNNLite未安装: pip3 install rknn-toolkit-lite2"
    echo "  - NumPy未安装: pip3 install numpy opencv-python-headless pillow"
    echo "  - NPU设备不存在: sudo modprobe rknpu"
    exit 1
fi
