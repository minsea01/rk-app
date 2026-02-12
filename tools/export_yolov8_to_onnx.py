#!/usr/bin/env python3
"""YOLOv8/YOLO11 model export to ONNX format.

This script exports Ultralytics YOLO models to ONNX format for deployment
and further conversion to RKNN format.
"""

import argparse
from pathlib import Path
import sys

# Import custom exceptions
from apps.exceptions import ModelLoadError, ConfigurationError
from apps.logger import setup_logger

# Setup logger
logger = setup_logger(__name__, level="INFO")


def export(
    weights: str,
    imgsz: int,
    opset: int,
    simplify: bool,
    dynamic: bool,
    half: bool,
    outdir: Path,
    outfile: str = None,
):
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ConfigurationError(
            f"Ultralytics not installed. Please run: pip install ultralytics\nError: {e}"
        ) from e

    # Validate weights file exists
    weights_path = Path(weights)
    if not weights_path.exists():
        raise ModelLoadError(f"Weights file not found: {weights}")

    logger.info(f"Loading model: {weights}")
    try:
        model = YOLO(weights)
    except (RuntimeError, ValueError, FileNotFoundError, Exception) as e:
        raise ModelLoadError(f"Failed to load model from {weights}: {e}") from e

    logger.info(f"Exporting to ONNX (imgsz={imgsz}, opset={opset}, simplify={simplify})")
    try:
        onnx_path = model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
            dynamic=dynamic,
            half=half,
        )
    except (RuntimeError, ValueError, TypeError) as e:
        raise ModelLoadError(f"Failed to export model to ONNX: {e}") from e

    # Move result into the outdir if ultralytics writes into CWD
    onnx_path = Path(onnx_path)
    target = outdir / (outfile if outfile else onnx_path.name)
    if onnx_path.resolve() != target.resolve():
        try:
            target.write_bytes(onnx_path.read_bytes())
        except (IOError, OSError, PermissionError) as e:
            raise ModelLoadError(f"Failed to move ONNX file to {target}: {e}") from e

    logger.info(f"Successfully exported ONNX: {target}")
    return target


def main():
    p = argparse.ArgumentParser(description="Export YOLOv8 to ONNX for RKNN conversion")
    p.add_argument("--weights", type=str, default="yolov8s.pt", help="YOLOv8 .pt weights")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--simplify", action="store_true", default=True)
    p.add_argument("--no-simplify", dest="simplify", action="store_false")
    p.add_argument("--dynamic", action="store_true", default=False)
    p.add_argument("--half", action="store_true", default=False)
    p.add_argument("--outdir", type=Path, default=Path("artifacts/models"))
    p.add_argument(
        "--outfile", type=str, default=None, help="output ONNX file name (e.g., best.onnx)"
    )
    args = p.parse_args()

    try:
        export(**vars(args))
        return 0
    except (ModelLoadError, ConfigurationError) as e:
        logger.error(f"Export failed: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
