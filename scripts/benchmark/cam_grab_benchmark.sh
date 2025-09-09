#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#  V4L2 UVC camera (/dev/video0) @ 1920x1080
#    ./scripts/cam_grab_benchmark.sh v4l2 0 1920 1080 30
#  GigE Vision via aravis (needs aravis):
#    ./scripts/cam_grab_benchmark.sh gige <device-id-or-first> 2048 1080 30

MODE=${1:-}
DEV=${2:-0}
WIDTH=${3:-1920}
HEIGHT=${4:-1080}
FPS=${5:-30}

if [ "$MODE" = "v4l2" ]; then
  echo "Benchmarking V4L2 /dev/video${DEV} ${WIDTH}x${HEIGHT}@${FPS}"
  gst-launch-1.0 -v v4l2src device=/dev/video${DEV} ! video/x-raw,width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1 ! fakesink sync=false
elif [ "$MODE" = "gige" ]; then
  echo "Benchmarking GigE Vision ${DEV} ${WIDTH}x${HEIGHT}@${FPS}"
  if ! gst-inspect-1.0 aravissrc >/dev/null 2>&1; then
    echo "aravissrc not found. Install aravis/gstreamer plugin."; exit 1
  fi
  # If DEV is not a full id, aravissrc will pick the first device
  gst-launch-1.0 -v aravissrc camera-name=${DEV} ! video/x-raw,width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1 ! fakesink sync=false
else
  echo "Usage: $0 v4l2 <video_index> <W> <H> <FPS> | gige <device_name_or_first> <W> <H> <FPS>"
  exit 1
fi

