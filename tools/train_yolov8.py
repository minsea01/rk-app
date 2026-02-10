#!/usr/bin/env python3
"""YOLOv8 training script using Ultralytics."""
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def parse_batch(value):
    """Parse Ultralytics batch argument.

    Supports:
      - integer batch size, e.g. 16
      - fractional auto-batch, e.g. 0.7
      - "auto" (mapped to -1)
    """
    if isinstance(value, (int, float)):
        return value

    s = str(value).strip().lower()
    if s == 'auto':
        return -1

    try:
        return int(s)
    except ValueError:
        pass

    try:
        return float(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"invalid --batch value '{value}', expected int/float/auto"
        ) from e


def main() -> int:
    """Main training entry point.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    ap = argparse.ArgumentParser(description='Train YOLOv8 (Ultralytics) for detection')
    ap.add_argument('--data', type=str, required=True, help='dataset YAML (Ultralytics/YOLO format)')
    ap.add_argument('--model', type=str, default='yolov8s.pt', help='initial weights or model yaml (e.g., yolov8s.yaml)')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=parse_batch, default=16, help='batch size (e.g. 16), fraction (e.g. 0.7), or auto')
    ap.add_argument('--device', type=str, default='0', help='GPU id, e.g., 0; use cpu for CPU')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--project', type=str, default='runs/train')
    ap.add_argument('--name', type=str, default='exp')
    ap.add_argument('--lr0', type=float, default=None)
    ap.add_argument('--lrf', type=float, default=None)
    ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        logger.error(f'Ultralytics not installed. pip install ultralytics. Error: {e}')
        return 2

    model = YOLO(args.model)

    train_args = {
        'data': args.data,
        'imgsz': args.imgsz,
        'epochs': args.epochs,
        'batch': args.batch,
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'patience': args.patience,
        'seed': args.seed,
    }
    # Optional hyper-params
    if args.lr0 is not None:
        train_args['lr0'] = args.lr0
    if args.lrf is not None:
        train_args['lrf'] = args.lrf

    logger.info(f'Training args: {train_args}')
    res = model.train(**train_args)
    logger.info(str(res))
    # Export best weights path
    best = Path(res.save_dir) / 'weights' / 'best.pt'
    logger.info(f'Best weights: {best} (exists={best.exists()})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
