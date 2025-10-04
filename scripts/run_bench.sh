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
# Wait for server to report its port
for i in {1..20}; do
  if grep -q "listening_port" "$ART_DIR/http_ingest.log"; then break; fi
  sleep 0.1
done
PORT=$(jq -r '.listening_port' "$ART_DIR/http_ingest.log" | head -n1)
sleep 0.3
python3 "$ROOT_DIR/tools/http_post.py" --url http://127.0.0.1:${PORT}/ingest --file "$ART_DIR/bench_summary.json" || echo "HTTP post skipped (receiver not ready)" >> "$ART_DIR/runner.log"
sleep 0.2
kill $SRV_PID || true
echo "Done. See artifacts/ for outputs." | tee -a "$ART_DIR/runner.log"
