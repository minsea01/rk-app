#!/usr/bin/env bash
set -euo pipefail

OUT_JSON=${1:-artifacts/ffprobe.json}
SAMPLE=${2:-artifacts/sample.mp4}

mkdir -p "$(dirname "$OUT_JSON")"
mkdir -p "$(dirname "$SAMPLE")"

# Generate a short 1080p@30fps sample
ffmpeg -v error -f lavfi -i testsrc=size=1920x1080:rate=30 -t 1 \
  -pix_fmt yuv420p -c:v libx264 -preset ultrafast -y "$SAMPLE" >/dev/null 2>&1

# Probe to JSON
ffprobe -v error -show_streams -show_format -print_format json "$SAMPLE" | tee "$OUT_JSON" >/dev/null

