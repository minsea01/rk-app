#!/usr/bin/env bash
ok(){ printf "[OK] %s\n" "$1"; }
miss(){ printf "[MISS] %s\n" "$1"; }

check_bin(){ command -v "$1" >/dev/null 2>&1 && ok "$1" || miss "$1"; }
check_gst(){
  gst-inspect-1.0 "$1" >/dev/null 2>&1 && ok "gst:$1" || miss "gst:$1"
}

echo "== Build / Cross / Debug =="
check_bin gcc
check_bin g++
check_bin cmake
check_bin ninja
check_bin pkg-config
check_bin qemu-aarch64
check_bin gdb-multiarch
check_bin aarch64-linux-gnu-gcc

echo "== Net tools =="
check_bin iperf3
check_bin tcpdump
check_bin tshark
check_bin ethtool

echo "== Time sync (PTP) =="
check_bin ptp4l
check_bin phc2sys

echo "== GStreamer core =="
check_bin gst-launch-1.0
check_bin gst-inspect-1.0
for p in coreelements udpsrc rtpjitterbuffer rtph264depay h264parse avdec_h264 \
         videoconvert appsink appsrc autovideosink ; do
  check_gst "$p"
done

echo "== Optional HW accel (按平台可选) =="
# x86 可检查 vaapi；RK3588 以后在板子上检查 rockchip 插件
check_gst vaapih264dec >/dev/null 2>&1
check_gst v4l2slh264dec >/dev/null 2>&1

echo "== OpenCV / Python 快速原型 =="
check_bin python3
python3 - <<'PY' >/dev/null 2>&1 && ok "python3: cv2/numpy/matplotlib" || miss "python3: cv2/numpy/matplotlib"
import cv2, numpy, matplotlib
PY

echo "== sysctl 建议 (仅提示) =="
for k in net.core.rmem_max net.core.rmem_default net.core.netdev_max_backlog; do
  printf "%s=%s\n" "$k" "$(sysctl -n $k 2>/dev/null || echo '?')"
done

