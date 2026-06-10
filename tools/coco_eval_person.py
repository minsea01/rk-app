"""Evaluate a single-class person YOLO model on COCO val2017 using pycocotools."""

import json
import sys
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

MODEL = sys.argv[1] if len(sys.argv) > 1 else "artifacts/models/yolov8s_person_aug.pt"
IMGSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 640
IMG_DIR = "/mnt/e/WSL/datasets/rk-app-datasets/coco/val2017"
ANN = "/mnt/e/WSL/datasets/rk-app-datasets/coco/annotations/instances_val2017.json"
OUT = Path("artifacts/coco_eval_preds.json")

print(f"Model: {MODEL}  imgsz={IMGSZ}")
model = YOLO(MODEL)

coco_gt = COCO(ANN)
person_img_ids = set(coco_gt.getImgIds(catIds=[1]))
all_img_ids = coco_gt.getImgIds()
print(f"COCO val2017: {len(all_img_ids)} images, {len(person_img_ids)} contain person")

predictions = []
results = model.predict(
    source=IMG_DIR,
    imgsz=IMGSZ,
    conf=0.001,
    iou=0.65,
    save=False,
    verbose=False,
    stream=True,
    device="cpu",
)

n = 0
for r in results:
    img_id = int(Path(r.path).stem)
    if r.boxes is None or len(r.boxes) == 0:
        n += 1
        continue
    for box, conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
        x1, y1, x2, y2 = box
        predictions.append(
            {
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": conf,
            }
        )
    n += 1
    if n % 500 == 0:
        print(f"  processed {n}/5000 ({len(predictions)} detections)")

print(f"Total: {n} images, {len(predictions)} detections")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(predictions))

if not predictions:
    print("No detections — skipping eval.")
    sys.exit(1)

coco_dt = coco_gt.loadRes(str(OUT))
evaluator = COCOeval(coco_gt, coco_dt, "bbox")
evaluator.params.catIds = [1]
evaluator.params.imgIds = list(person_img_ids)
evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()
