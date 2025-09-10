Deploy kit for RK3588

Contents
- best.onnx: exported ONNX (640, opset=12)
- dataset.txt: ~200 calibration images (absolute paths)
- rknn_build.py: ONNX -> RKNN (INT8) builder
- run_cpu.sh: CPU fallback prediction (for smoke test)

Usage on PC
1) CPU smoke test
   ./run_cpu.sh

2) (Optional) Validate ONNX locally
   yolo val model=best.onnx imgsz=640 data=../industrial_dataset/data.yaml

Copy to board
scp -r deploy_rk3588/ BOARD_USER@BOARD_IP:~/ai/

On board: quantize to RKNN
cd ~/ai/deploy_rk3588
python3 rknn_build.py --onnx best.onnx --dataset dataset.txt --out best_int8.rknn

Notes
- Ensure rknn-toolkit2 installed on PC if converting there.
- For live camera pipeline, integrate RGA letterbox and RKNPU2 runtime in C++.


