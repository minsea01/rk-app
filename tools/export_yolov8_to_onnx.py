#!/usr/bin/env python3
import argparse
from pathlib import Path

def export(weights: str, imgsz: int, opset: int, simplify: bool, dynamic: bool, half: bool, outdir: Path, outfile: str = None):
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise SystemExit(f"Ultralytics not installed. pip install ultralytics. Error: {e}")

    model = YOLO(weights)
    onnx_path = model.export(
        format='onnx',
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic,
        half=half,
    )
    # Move result into the outdir if ultralytics writes into CWD
    onnx_path = Path(onnx_path)
    target = outdir / (outfile if outfile else onnx_path.name)
    if onnx_path.resolve() != target.resolve():
        target.write_bytes(onnx_path.read_bytes())
    print(f"Exported ONNX: {target}")
    return target


def main():
    p = argparse.ArgumentParser(description='Export YOLOv8 to ONNX for RKNN conversion')
    p.add_argument('--weights', type=str, default='yolov8s.pt', help='YOLOv8 .pt weights')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--opset', type=int, default=12)
    p.add_argument('--simplify', action='store_true', default=True)
    p.add_argument('--no-simplify', dest='simplify', action='store_false')
    p.add_argument('--dynamic', action='store_true', default=False)
    p.add_argument('--half', action='store_true', default=False)
    p.add_argument('--outdir', type=Path, default=Path('artifacts/models'))
    p.add_argument('--outfile', type=str, default=None, help='output ONNX file name (e.g., best.onnx)')
    args = p.parse_args()
    export(**vars(args))

if __name__ == '__main__':
    main()
