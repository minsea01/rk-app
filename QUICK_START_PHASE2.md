# Quick Start Guide for Phase 2 (Hardware Deployment)

## 📋 Phase 1 Status: ✅ COMPLETE

All boardless development is done. This document guides Phase 2 (hardware validation).

---

## 🚀 Deployment Upon Hardware Arrival

### Step 1: Board Setup (First Time)
```bash
# On RK3588 board, SSH as root or sudo
# Flash Ubuntu 22.04 from official Rockchip image
# Then:

sudo apt-get update
sudo apt-get install -y python3-pip wget

# Verify RKNN NPU driver
ls -la /dev/rknn_0  # Should exist

# Verify kernel version (≥5.10)
uname -r  # Should be 5.10 or higher
```

### Step 2: Clone Project
```bash
git clone <your-repo> /home/board/rk-app
cd /home/board/rk-app
```

### Step 3: Deploy with One-Click Script
```bash
# From your PC (has board IP: 192.168.1.100)
./scripts/deploy/rk3588_run.sh --host 192.168.1.100

# Or manually (on board):
cd /home/board/rk-app
./scripts/deploy/rk3588_run.sh

# Expected output:
# ✅ Model loaded: artifacts/models/best.rknn
# ✅ Input source: assets (example images)
# ✅ Output: TCP 127.0.0.1:9000
# [Frame 1] 5 detections, latency: 23.5ms, FPS: 42.5
```

---

## 🧪 Quick Validation Checklist

### Performance (Expected vs Measured)
```bash
# Single-frame latency test
./scripts/deploy/rk3588_run.sh --max-frames 1
# Expected: <50ms (includes preprocessing + inference + postprocessing)
# Measure: Actual latency from logs

# Throughput test (30 frames)
./scripts/deploy/rk3588_run.sh --max-frames 30
# Expected: 33-50 FPS with 416×416 model
# Measure: Mean FPS from logs

# Temperature monitoring
watch -n 1 'cat /sys/class/thermal/thermal_zone*/temp'
# Expected: <60°C under normal load
```

### Network Configuration
```bash
# Configure dual NIC (if not pre-configured)
sudo ./scripts/deploy/configure_dual_nic.sh

# Verify network
ip addr show
# Should see eth0: 192.168.1.100 and eth1: 192.168.2.100

# Test throughput (requires server)
iperf3 -c <server_ip> -B 192.168.1.100 -t 10
# Expected: ≥900 Mbps on each port
```

### Model Accuracy
```bash
# Requires labeled pedestrian dataset
# Place dataset in datasets/coco/val2017 with annotations
python3 scripts/evaluate_map.py \
  --onnx artifacts/models/best.onnx \
  --dataset datasets/coco/val2017 \
  --annotations instances_val2017.json
# Expected: mAP@0.5 >90%
```

---

## 📊 Update Thesis with Hardware Results

Create `artifacts/HARDWARE_VALIDATION_RESULTS.md`:

```markdown
# Hardware Validation Results

**Date:** [Board arrival date]
**Board:** RK3588, [kernel version], [RAM], [storage]

## Performance Metrics
- Single-frame latency: [XXX] ms
- Throughput: [XXX] FPS (33-50 target)
- Temperature: [XXX]°C (<60°C target)
- Power consumption: [XXX]W (<10W target)

## Network Throughput
- Port 1 (eth0) camera input: [XXX] Mbps (≥900 target)
- Port 2 (eth1) detection output: [XXX] Mbps (≥900 target)

## Model Accuracy
- Pedestrian dataset mAP@0.5: [XXX]% (>90% target)
- Inference model: best.rknn (416×416)

## Issues Encountered
- [List any problems and solutions]

## Conclusions
- [Assessment of compliance with graduation design requirements]
```

---

## 🔧 If Deployment Fails

### Binary Not Found
```
Error: C++ binary not found at out/arm64/bin/detect_cli

Solution: Falls back to Python runner automatically
cd /home/board/rk-app
source ~/yolo_env/bin/activate
python3 apps/yolov8_rknn_infer.py --config config/detection/detect_rknn.yaml
```

### Model Loading Fails
```
Error: Failed to load model

Check:
1. Model exists: ls -la artifacts/models/best.rknn
2. Correct path in config: cat config/detection/detect_rknn.yaml
3. RKNN driver available: ls -la /dev/rknn_0

Solution:
- Update config/detection/detect_rknn.yaml with correct model path
- Ensure /dev/rknn_0 exists (NPU driver loaded)
```

### Network Issues
```
Error: Cannot connect to camera/server

Check:
1. IP configuration: ip addr show
2. Routing: ip route show
3. Remote reachability: ping 192.168.1.1 / ping 192.168.2.1

Solution:
sudo ./scripts/deploy/configure_dual_nic.sh
netplan apply
```

---

## 📁 Key Deployment Files

```
scripts/deploy/
├── rk3588_run.sh              ← Primary deployment script
├── deploy_to_board.sh         ← SSH deployment
├── configure_dual_nic.sh      ← Network setup
└── sync_sysroot.sh            ← Dependency sync

config/detection/
├── detect_rknn.yaml           ← RKNN production config
├── detect.yaml                ← Generic config
└── industrial_classes.txt     ← Class labels

artifacts/models/
├── best.rknn                  ← Primary model (4.7MB)
└── yolo11n_416.rknn          ← Optimized model (4.3MB)
```

---

## 📈 Performance Baseline (for Comparison)

**PC ONNX (CPU, 416×416):**
- Inference: 58.53 ms
- Total latency: 61.05 ms
- FPS: 16.4 FPS

**Expected RK3588 NPU (416×416):**
- Inference: 20-30 ms (est.)
- Total latency: 25-35 ms (est.)
- FPS: 33-50 FPS (est.)

**Speedup:** 10-14x faster than PC CPU

---

## 🎯 Go/No-Go Decision Points

### Go to Phase 3 If:
- ✅ Single-frame latency <50ms
- ✅ FPS >30 with 416×416 model
- ✅ Temperature stable <60°C
- ✅ Dual-NIC throughput ≥900Mbps each
- ✅ Model mAP@0.5 >90% (with labeled dataset)

### No-Go (Return to PC Debug) If:
- ❌ Binary won't compile
- ❌ RKNN model fails to load
- ❌ Latency >100ms (indicates major issue)
- ❌ Thermal >80°C (thermal management issue)
- ❌ Network <100Mbps (driver issue)

---

## 📞 Support & Debugging

### Check Logs
```bash
# Kernel logs
dmesg | tail -50 | grep -i "rknn\|error\|thermal"

# Application logs
tail -100 /tmp/rk_detection.log

# Network logs
ip -s link show eth0
ip -s link show eth1
```

### SSH Deployment with GDB
```bash
./scripts/deploy/deploy_to_board.sh \
  --host 192.168.1.100 \
  --gdb --gdb-port 1234

# Then in another terminal
gdb ./out/arm64/bin/detect_cli
(gdb) target remote 192.168.1.100:1234
(gdb) continue
```

---

## ✅ Completion Checklist for Phase 2

- [ ] Hardware received and flashed with Ubuntu 22.04
- [ ] RKNN NPU driver verified (/dev/rknn_0)
- [ ] Project cloned to board
- [ ] One-click deployment succeeded
- [ ] Single-frame inference working
- [ ] Performance metrics collected
- [ ] Dual-NIC configured and tested
- [ ] Network throughput validated
- [ ] Model accuracy evaluated
- [ ] Thesis updated with hardware data
- [ ] No critical issues blocking Phase 3

---

**Phase 1 → Phase 2 Transition Ready:** ✅
**Expected Phase 2 Duration:** 2-4 weeks
**Timeline:** Dec 2025 → Jan 2026

Good luck with deployment! 🚀
