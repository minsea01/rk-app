#!/usr/bin/env python3
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description='Train YOLOv8 (Ultralytics) for detection')
    ap.add_argument('--data', type=str, required=True, help='dataset YAML (Ultralytics/YOLO format)')
    ap.add_argument('--model', type=str, default='yolov8s.pt', help='initial weights or model yaml (e.g., yolov8s.yaml)')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=16)  # Fixed: int type, not str
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
    except Exception as e:
        raise SystemExit(f'Ultralytics not installed. pip install ultralytics. Error: {e}')

    model = YOLO(args.model)

    # Convert args to dict and ensure correct types
    train_args = {
        'data': str(args.data),
        'imgsz': int(args.imgsz), 
        'epochs': int(args.epochs),
        'batch': int(args.batch),  # Ensure integer type
        'device': str(args.device),
        'workers': int(args.workers),
        'project': str(args.project),
        'name': str(args.name),
        'patience': int(args.patience),
        'seed': int(args.seed),
    }
    
    # Optional hyper-params
    if args.lr0 is not None:
        train_args['lr0'] = float(args.lr0)
    if args.lrf is not None:
        train_args['lrf'] = float(args.lrf)

    print('Training args:', train_args)
    print('Batch type:', type(train_args['batch']))
    
    res = model.train(**train_args)
    print(res)
    
    # Export best weights path
    best = Path(res.save_dir) / 'weights' / 'best.pt'
    print(f'Best weights: {best} (exists={best.exists()})')


if __name__ == '__main__':
    main()