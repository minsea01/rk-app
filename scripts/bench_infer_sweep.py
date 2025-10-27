import argparse
import os
from typing import List
from pathlib import Path

from ultralytics import YOLO
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "bench_conf_sweep"


def parse_conf_list(conf_list_str: str) -> List[float]:
    return [float(x) for x in conf_list_str.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep confidence thresholds for YOLO val and report P/R/F1/mAP.")
    parser.add_argument("--model", required=True, help="Path to model .pt file")
    parser.add_argument("--data", required=True, help="Path to dataset YAML")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for validation")
    parser.add_argument("--device", default="0", help="CUDA device, e.g. 0 or 0,1 or cpu")
    parser.add_argument("--half", action="store_true", help="Use half precision (FP16)")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--conf_list", default="0.30,0.40,0.50,0.60,0.65,0.70,0.75", help="Comma-separated confidences to sweep")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR), help="Directory to save sweep results")
    parser.add_argument("--name", default="conf_sweep", help="Subdirectory name for this run")
    args = parser.parse_args()

    confs = parse_conf_list(args.conf_list)

    model = YOLO(args.model)

    out_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(out_dir, exist_ok=True)
    tsv_path = os.path.join(out_dir, "results.tsv")

    print("conf\tP\tR\tF1\tmAP50\tmAP50-95")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("conf\tP\tR\tF1\tmAP50\tmAP50-95\n")

    best_conf = None
    best_f1 = -1.0

    for conf in confs:
        metrics = model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            half=args.half,
            conf=conf,
            iou=args.iou,
            verbose=False,
            save=False,
            plots=False,
        )

        # metrics.box fields: p, r are per-class arrays; map50/map are scalar
        precision = float(np.mean(metrics.box.p)) if hasattr(metrics.box, "p") else float("nan")
        recall = float(np.mean(metrics.box.r)) if hasattr(metrics.box, "r") else float("nan")
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else float("nan")
        map50 = float(metrics.box.map50)
        map5095 = float(metrics.box.map)

        line = f"{conf:.2f}\t{precision:.3f}\t{recall:.3f}\t{f1:.3f}\t{map50:.3f}\t{map5095:.3f}"
        print(line)
        with open(tsv_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        if not np.isnan(f1) and f1 > best_f1:
            best_f1 = f1
            best_conf = conf

    if best_conf is not None:
        best_path = os.path.join(out_dir, "best.txt")
        msg = f"best_conf={best_conf:.2f} best_F1={best_f1:.3f}"
        print(msg)
        with open(best_path, "w", encoding="utf-8") as f:
            f.write(msg + "\n")
    else:
        print("best_conf=N/A best_F1=N/A")


if __name__ == "__main__":
    main()


