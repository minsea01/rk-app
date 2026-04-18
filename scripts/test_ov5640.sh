#!/bin/bash
# ============================================================
# OV5640 板端就绪性检查脚本
# 在 RK3588 上执行，验证 MIPI CSI 摄像头各环节是否正常
# ============================================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; exit 1; }
warn() { echo -e "${YELLOW}!${NC} $1"; }

echo "========================================"
echo "  OV5640 + RK3588 就绪性检查"
echo "========================================"

# 1. 驱动加载检查
echo
echo "[1/6] 检查内核驱动..."
dmesg | grep -qi "ov5640.*[Dd]etected\|ov5640.*probe" \
  && ok "OV5640 驱动已加载" \
  || warn "dmesg 未检测到 OV5640（可能板子启动太久，dmesg 已被冲掉；可继续）"

dmesg | grep -qi "rkcif\|rkisp" && ok "RKCIF/RKISP 驱动就绪" \
  || warn "未检测到 rkcif/rkisp"

# 2. v4l2 设备节点
echo
echo "[2/6] 检查 /dev/video* 节点..."
if ls /dev/video* &>/dev/null; then
  ok "发现 v4l2 设备节点:"
  ls -l /dev/video*
else
  fail "没有任何 /dev/video* 节点 — 需修改设备树启用 MIPI CSI"
fi

# 3. v4l2-ctl 能力查询
echo
echo "[3/6] 查询 OV5640 支持的格式..."
command -v v4l2-ctl >/dev/null || fail "v4l2-ctl 未安装: sudo apt install v4l-utils"

for dev in /dev/video{0,1,11,12}; do
  [ -e "$dev" ] || continue
  echo "--- $dev ---"
  v4l2-ctl -d "$dev" --list-formats-ext 2>/dev/null | head -20 || true
done

# 4. GStreamer 取流 1 帧
echo
echo "[4/6] GStreamer 取流测试（抓 1 帧存 /tmp/ov5640_test.jpg）..."
command -v gst-launch-1.0 >/dev/null || fail "gst-launch-1.0 未安装: sudo apt install gstreamer1.0-tools"

rm -f /tmp/ov5640_test.jpg
if timeout 10 gst-launch-1.0 -e v4l2src device=/dev/video0 num-buffers=1 \
     ! videoconvert ! jpegenc ! filesink location=/tmp/ov5640_test.jpg >/dev/null 2>&1; then
  [ -s /tmp/ov5640_test.jpg ] \
    && ok "成功抓帧 ($(stat -c%s /tmp/ov5640_test.jpg) bytes)" \
    || fail "抓帧文件为空"
else
  warn "v4l2src 抓帧失败，尝试 videotestsrc 验证 GStreamer 本身"
  gst-launch-1.0 videotestsrc num-buffers=1 ! jpegenc ! filesink location=/tmp/gst_test.jpg >/dev/null 2>&1 \
    && warn "GStreamer 正常，说明问题在 v4l2src/设备树" \
    || fail "GStreamer 本身异常"
fi

# 5. 1080p@30fps 持续采集 3 秒（测稳定性）
echo
echo "[5/6] 1080p@30fps 稳定性测试（持续 3 秒）..."
if timeout 5 gst-launch-1.0 v4l2src device=/dev/video0 num-buffers=90 \
     ! "video/x-raw,width=1920,height=1080,framerate=30/1" \
     ! fakesink >/dev/null 2>&1; then
  ok "1080p@30fps 稳定采集 90 帧成功"
else
  warn "1080p@30fps 有丢帧或不支持，需降级到 720p 或调整格式"
fi

# 6. RKNN 运行时 + RGA 库检查
echo
echo "[6/6] 检查运行时依赖..."
[ -e /usr/lib/librknnrt.so ] || [ -e /usr/lib/aarch64-linux-gnu/librknnrt.so ] \
  && ok "librknnrt.so 存在" || warn "RKNN Runtime 未安装"

[ -e /usr/lib/librga.so ] || [ -e /usr/lib/aarch64-linux-gnu/librga.so ] \
  && ok "librga.so 存在（可启用 RGA 加速）" || warn "librga 未安装"

echo
echo "========================================"
echo "  ✅ OV5640 就绪性检查完成"
echo "  下一步: ./build/arm64-release/detect_cli \\"
echo "          --cfg config/detection/detect_ov5640.yaml"
echo "========================================"
