#!/usr/bin/env bash
set -euo pipefail

# 双网卡流水线脚本：网口1接收相机流 → 推理 → 网口2上传结果
# 用于验证毕设双千兆以太网要求

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 默认参数
INPUT_INTERFACE="eth0"
INPUT_PORT="8554"        # RTSP默认端口
OUTPUT_INTERFACE="eth1"
OUTPUT_HOST=""
OUTPUT_PORT="8080"
MODEL="$ROOT_DIR/artifacts/models/yolo11n_416.rknn"
IMGSZ=416
CONF=0.5
FORMAT="json"            # json | udp

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --input-interface <eth>     Input network interface (default: eth0)
  --input-port <port>         Input port for camera stream (default: 8554)
  --input-source <url>        Direct camera source URL (overrides interface)
  --output-interface <eth>    Output network interface (default: eth1)
  --output-host <ip>          Target host for detection results (required)
  --output-port <port>        Target port (default: 8080)
  --model <path>              RKNN model path (default: artifacts/models/yolo11n_416.rknn)
  --imgsz <size>              Input image size (default: 416)
  --conf <threshold>          Confidence threshold (default: 0.5)
  --format <json|udp>         Output format (default: json)
  -h, --help                  Show this help

Examples:
  # Camera on eth0 (192.168.1.x), server on eth1 (192.168.2.x)
  $0 --input-interface eth0 --output-host 192.168.2.200 --output-port 8080

  # RTSP camera with custom URL
  $0 --input-source rtsp://192.168.1.100:8554/stream --output-host 192.168.2.200

  # UDP output for low latency
  $0 --input-interface eth0 --output-host 192.168.2.200 --format udp
EOF
}

# Parse arguments
INPUT_SOURCE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-interface) INPUT_INTERFACE="$2"; shift 2;;
        --input-port) INPUT_PORT="$2"; shift 2;;
        --input-source) INPUT_SOURCE="$2"; shift 2;;
        --output-interface) OUTPUT_INTERFACE="$2"; shift 2;;
        --output-host) OUTPUT_HOST="$2"; shift 2;;
        --output-port) OUTPUT_PORT="$2"; shift 2;;
        --model) MODEL="$2"; shift 2;;
        --imgsz) IMGSZ="$2"; shift 2;;
        --conf) CONF="$2"; shift 2;;
        --format) FORMAT="$2"; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown option: $1"; usage; exit 1;;
    esac
done

if [[ -z "$OUTPUT_HOST" ]]; then
    echo "❌ --output-host is required"
    usage
    exit 1
fi

# Check model exists
if [[ ! -f "$MODEL" ]]; then
    echo "❌ Model not found: $MODEL"
    exit 1
fi

echo "=========================================="
echo "Dual NIC Pipeline Configuration"
echo "=========================================="
echo "Input Interface:  $INPUT_INTERFACE"
echo "Input Port:       $INPUT_PORT"
if [[ -n "$INPUT_SOURCE" ]]; then
    echo "Input Source:     $INPUT_SOURCE"
fi
echo "Output Interface: $OUTPUT_INTERFACE"
echo "Output Host:      $OUTPUT_HOST:$OUTPUT_PORT"
echo "Model:            $MODEL"
echo "Image Size:       ${IMGSZ}x${IMGSZ}"
echo "Confidence:       $CONF"
echo "Output Format:    $FORMAT"
echo "=========================================="
echo ""

# Verify interfaces exist
if ! ip link show "$INPUT_INTERFACE" >/dev/null 2>&1; then
    echo "❌ Input interface not found: $INPUT_INTERFACE"
    echo "Available interfaces:"
    ip link show | grep -E "^[0-9]+" | awk '{print $2}' | sed 's/:$//'
    exit 1
fi

if ! ip link show "$OUTPUT_INTERFACE" >/dev/null 2>&1; then
    echo "❌ Output interface not found: $OUTPUT_INTERFACE"
    exit 1
fi

# Get interface IPs
INPUT_IP=$(ip -4 addr show "$INPUT_INTERFACE" | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -1)
OUTPUT_IP=$(ip -4 addr show "$OUTPUT_INTERFACE" | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -1)

echo "Network Configuration:"
echo "  $INPUT_INTERFACE:  $INPUT_IP (camera input)"
echo "  $OUTPUT_INTERFACE: $OUTPUT_IP (detection output)"
echo ""

# Construct input source URL
if [[ -z "$INPUT_SOURCE" ]]; then
    # Auto-detect camera stream (assuming RTSP on input network)
    # User should replace this with actual camera IP
    echo "⚠️  No --input-source specified. Please provide camera stream URL."
    echo "   Example: rtsp://192.168.1.100:8554/stream"
    echo ""
    echo "For testing, will use test image instead."
    INPUT_SOURCE="$ROOT_DIR/assets/test.jpg"
fi

# Create Python streaming script
STREAM_SCRIPT="/tmp/dual_nic_stream_$$.py"
cat > "$STREAM_SCRIPT" << PYEOF
#!/usr/bin/env python3
import sys
import time
import json
import socket
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, "$ROOT_DIR")

from apps.logger import setup_logger
from apps.utils.preprocessing import preprocess_board

logger = setup_logger(__name__)

# Configuration
INPUT_SOURCE = "$INPUT_SOURCE"
OUTPUT_HOST = "$OUTPUT_HOST"
OUTPUT_PORT = $OUTPUT_PORT
OUTPUT_IP = "$OUTPUT_IP"  # bind TX to上传网口
MODEL_PATH = "$MODEL"
IMGSZ = $IMGSZ
CONF = $CONF
FORMAT = "$FORMAT"

def bind_if_possible(sock):
    """Bind socket to output interface IP if present to force eth1 egress."""
    if OUTPUT_IP:
        try:
            sock.bind((OUTPUT_IP, 0))
        except OSError as e:
            logger.warning(f"Bind to {OUTPUT_IP} failed: {e}")

def send_udp(data, host, port):
    """Send detection results via UDP"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    bind_if_possible(sock)
    payload = json.dumps(data).encode('utf-8')
    sock.sendto(payload, (host, port))
    sock.close()

def send_tcp(data, host, port):
    """Send detection results via TCP"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    bind_if_possible(sock)
    try:
        sock.connect((host, port))
        payload = json.dumps(data).encode('utf-8')
        sock.sendall(payload + b'\n')
        sock.close()
        return True
    except Exception as e:
        logger.error(f"TCP send failed: {e}")
        return False

def main():
    # Load RKNNLite
    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        logger.error("RKNNLite not installed")
        sys.exit(1)

    logger.info(f"Loading RKNN model: {MODEL_PATH}")
    rknn = RKNNLite()
    ret = rknn.load_rknn(MODEL_PATH)
    if ret != 0:
        logger.error(f"Load model failed: ret={ret}")
        sys.exit(1)

    ret = rknn.init_runtime(core_mask=0x7)
    if ret != 0:
        logger.error(f"Init runtime failed: ret={ret}")
        sys.exit(1)

    logger.info(f"Opening video source: {INPUT_SOURCE}")

    # Open video stream
    if INPUT_SOURCE.startswith("rtsp://") or INPUT_SOURCE.startswith("http://"):
        cap = cv2.VideoCapture(INPUT_SOURCE)
    elif INPUT_SOURCE.isdigit():
        cap = cv2.VideoCapture(int(INPUT_SOURCE))
    else:
        # Single image mode
        cap = None

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Read frame
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame, retrying...")
                    time.sleep(0.1)
                    continue
            else:
                # Single image mode
                frame = cv2.imread(INPUT_SOURCE)
                if frame is None:
                    logger.error(f"Failed to load image: {INPUT_SOURCE}")
                    break

            # Preprocess
            input_data = preprocess_board(frame, target_size=IMGSZ)

            # Inference
            t0 = time.perf_counter()
            outputs = rknn.inference(inputs=[input_data])
            t1 = time.perf_counter()
            inference_ms = (t1 - t0) * 1000

            # Simple detection count (without full postprocessing)
            output = outputs[0][0]
            if output.shape[0] == 84:
                output = output.T

            scores = output[:, 4:].max(axis=1)
            detections = int((scores >= CONF).sum())

            # Prepare results
            result = {
                "timestamp": time.time(),
                "frame_id": frame_count,
                "detections": detections,
                "inference_ms": round(inference_ms, 2),
                "fps": round(1000.0 / inference_ms, 2) if inference_ms > 0 else 0,
            }

            # Send via network
            if FORMAT == "udp":
                send_udp(result, OUTPUT_HOST, OUTPUT_PORT)
            else:
                send_tcp(result, OUTPUT_HOST, OUTPUT_PORT)

            frame_count += 1

            # Print progress
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                logger.info(f"Frame {frame_count}: {detections} detections, "
                           f"{inference_ms:.1f}ms inference, {fps:.1f} FPS overall")

            # Single image mode: exit after one frame
            if cap is None:
                logger.info("Single image mode: completed")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if cap is not None:
            cap.release()
        rknn.release()

        elapsed = time.time() - start_time
        if elapsed > 0:
            logger.info(f"Processed {frame_count} frames in {elapsed:.1f}s "
                       f"({frame_count/elapsed:.1f} FPS)")

if __name__ == "__main__":
    main()
PYEOF

chmod +x "$STREAM_SCRIPT"

# Run the streaming script
echo "Starting dual NIC pipeline..."
echo "Press Ctrl+C to stop"
echo ""

python3 "$STREAM_SCRIPT"

# Cleanup
rm -f "$STREAM_SCRIPT"

echo ""
echo "Pipeline stopped."
