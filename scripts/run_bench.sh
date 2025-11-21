#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ART_DIR="$ROOT_DIR/artifacts"
mkdir -p "$ART_DIR"

echo "[1/4] iperf3 bench (loopback)" | tee "$ART_DIR/runner.log"
bash "$ROOT_DIR/tools/iperf3_bench.sh" "$ART_DIR/iperf3.json"

echo "[2/4] ffprobe sample (1080p@30fps)" | tee -a "$ART_DIR/runner.log"
bash "$ROOT_DIR/tools/ffprobe_probe.sh" "$ART_DIR/ffprobe.json" "$ART_DIR/sample.mp4"

echo "[3/4] aggregate results" | tee -a "$ART_DIR/runner.log"
python3 "$ROOT_DIR/tools/aggregate.py" \
  --iperf3 "$ART_DIR/iperf3.json" \
  --ffprobe "$ART_DIR/ffprobe.json" \
  --out-json "$ART_DIR/bench_summary.json" \
  --out-csv "$ART_DIR/bench_summary.csv" \
  --out-md "$ART_DIR/bench_report.md"

echo "[4/4] HTTP ingest validation" | tee -a "$ART_DIR/runner.log"
python3 "$ROOT_DIR/tools/http_receiver.py" --port 0 \
  > "$ART_DIR/http_ingest.log" 2>&1 &
SRV_PID=$!

# Fixed: Wait for server to report its port AND verify server is accepting connections
PORT=""
for i in {1..50}; do
  if grep -q "listening_port" "$ART_DIR/http_ingest.log" 2>/dev/null; then
    PORT=$(jq -r '.listening_port' "$ART_DIR/http_ingest.log" 2>/dev/null | head -n1)
    # Verify port is valid
    if [[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT > 0 && PORT < 65536 )); then
      # Test TCP connection (robust check)
      if timeout 0.2 bash -c "echo > /dev/tcp/127.0.0.1/$PORT" 2>/dev/null; then
        echo "HTTP server ready on port $PORT" | tee -a "$ART_DIR/runner.log"
        break
      fi
    fi
  fi
  sleep 0.1
  # Timeout after 5 seconds
  if (( i == 50 )); then
    echo "⚠️  HTTP server failed to start within 5 seconds" | tee -a "$ART_DIR/runner.log"
    kill $SRV_PID 2>/dev/null || true
    PORT=""
  fi
done

# Only attempt POST if server is confirmed ready
if [[ -n "$PORT" ]]; then
  python3 "$ROOT_DIR/tools/http_post.py" --url "http://127.0.0.1:${PORT}/ingest" --file "$ART_DIR/bench_summary.json" || \
    echo "⚠️  HTTP post failed" | tee -a "$ART_DIR/runner.log"
  sleep 0.2
  kill $SRV_PID 2>/dev/null || true
else
  echo "⚠️  HTTP post skipped (receiver not ready)" | tee -a "$ART_DIR/runner.log"
fi

echo "Done. See artifacts/ for outputs." | tee -a "$ART_DIR/runner.log"
