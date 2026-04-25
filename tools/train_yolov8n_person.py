"""Train YOLOv8n on the same person dataset as yolov8s_person_aug for apples-to-apples comparison."""

from ultralytics.models.yolo.model import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="datasets/person_aug.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=4,
    optimizer="SGD",
    project="runs/detect",
    name="yolov8n_person_aug",
    exist_ok=True,
    patience=50,
    save=True,
    plots=True,
    seed=0,
    deterministic=True,
    close_mosaic=10,
    cos_lr=False,
    verbose=True,
)

print("=" * 60)
print("Training complete.")
if results is not None and hasattr(results, "results_dict"):
    d = results.results_dict
    print(f"Best mAP50      = {d.get('metrics/mAP50(B)', 'N/A')}")
    print(f"Best mAP50-95   = {d.get('metrics/mAP50-95(B)', 'N/A')}")
