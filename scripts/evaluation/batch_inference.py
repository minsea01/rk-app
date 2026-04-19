#!/usr/bin/env python3
"""批量推理脚本 - 使用YOLOv8n Person模型"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

sys.path.insert(0, str(REPO_ROOT))

from apps.utils.yolo_post import letterbox, sigmoid, nms


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def batch_inference(image_paths, model_path, conf_thres=0.6, iou_thres=0.65, max_images=10):
    """批量推理多张图片"""

    model_path = Path(model_path)

    print("=" * 70)
    print("YOLOv8n Person 批量推理")
    print("=" * 70)
    print("模型:", _display_path(model_path))
    print("置信度阈值:", conf_thres)
    print("NMS阈值:", iou_thres)
    print("图片数量:", min(len(image_paths), max_images))
    print("=" * 70)

    # 加载模型
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    imgsz = 640

    results = []
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(image_paths[:max_images]):
        img_name = Path(img_path).name
        print(f"\n[{idx + 1}/{min(len(image_paths), max_images)}] 处理: {img_name}")

        # 加载图像
        img = cv2.imread(str(img_path))
        if img is None:
            print("   [错误] 无法读取图像")
            continue

        orig_h, orig_w = img.shape[:2]
        print(f"   尺寸: {orig_w}x{orig_h}")

        # 预处理
        start_total = time.time()
        img_resized, ratio, (dw, dh) = letterbox(img, imgsz)
        img_input = img_resized.transpose(2, 0, 1)
        img_input = np.expand_dims(img_input, 0).astype(np.float32) / 255.0

        # 推理
        start_infer = time.time()
        outputs = session.run(None, {input_name: img_input})
        infer_time = (time.time() - start_infer) * 1000

        # 后处理
        pred = outputs[0].transpose(0, 2, 1)[0]
        bbox_pred = pred[:, :4]
        cls_pred = pred[:, 4:]
        cls_scores = sigmoid(cls_pred)
        confs = cls_scores.max(axis=1)
        class_ids = cls_scores.argmax(axis=1)

        # 过滤
        person_mask = (confs >= conf_thres) & (class_ids == 0)
        valid_bbox = bbox_pred[person_mask]
        valid_confs = confs[person_mask]

        output_path = ARTIFACTS_DIR / f"batch_result_{idx + 1:03d}_{img_name}"
        output_label = _display_path(output_path)

        if len(valid_bbox) > 0:
            cx, cy, w, h = valid_bbox[:, 0], valid_bbox[:, 1], valid_bbox[:, 2], valid_bbox[:, 3]
            boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

            # 面积过滤
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            area_mask = (areas > 500) & (areas < 100000)
            boxes = boxes[area_mask]
            valid_confs = valid_confs[area_mask]

            if len(boxes) > 0:
                keep = nms(boxes, valid_confs, iou_thres=iou_thres)
                final_boxes = boxes[keep]
                final_confs = valid_confs[keep]

                # 绘制结果
                for box, conf in zip(final_boxes, final_confs):
                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(orig_w, x2), min(orig_h, y2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        img,
                        f"person {conf:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

                # 添加统计
                cv2.putText(
                    img,
                    f"{len(final_boxes)} persons | {infer_time:.1f}ms",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

                cv2.imwrite(str(output_path), img)

                total_time = (time.time() - start_total) * 1000
                print(
                    f"   [成功] 检测到 {len(final_boxes)} 人 | 推理: {infer_time:.1f}ms | "
                    f"总计: {total_time:.1f}ms"
                )

                results.append(
                    {
                        "name": img_name,
                        "persons": len(final_boxes),
                        "infer_time": infer_time,
                        "output": output_label,
                    }
                )
            else:
                print("   [过滤] 面积过滤后无目标")
                cv2.putText(
                    img,
                    "No person detected (size filter)",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                cv2.imwrite(str(output_path), img)
                results.append(
                    {
                        "name": img_name,
                        "persons": 0,
                        "infer_time": infer_time,
                        "output": output_label,
                    }
                )
        else:
            print("   [无目标] 未检测到 person")
            cv2.putText(
                img,
                "No person detected",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            cv2.imwrite(str(output_path), img)
            results.append(
                {
                    "name": img_name,
                    "persons": 0,
                    "infer_time": infer_time,
                    "output": output_label,
                }
            )

    # 汇总
    print("\n" + "=" * 70)
    print("批量推理汇总")
    print("=" * 70)
    print(f"{'图片名':<25} | {'人数':>6} | {'耗时(ms)':>10}")
    print("-" * 70)

    total_persons = 0
    total_time = 0
    for r in results:
        name = r["name"][:24]
        print(f'{name:<25} | {r["persons"]:>6} | {r["infer_time"]:>10.2f}')
        total_persons += r["persons"]
        total_time += r["infer_time"]

    print("-" * 70)
    if results:
        avg_time = total_time / len(results)
        print(f"总计: {len(results)}张图, {total_persons}人, 平均{avg_time:.2f}ms/张")
    print("=" * 70)

    return results


if __name__ == "__main__":
    # 收集图片
    image_paths = []

    assets_dir = REPO_ROOT / "assets"
    if assets_dir.exists():
        image_paths.extend(assets_dir.glob("*.jpg"))
        image_paths.extend(assets_dir.glob("*.png"))

    calib_dir = REPO_ROOT / "datasets/coco/calib_images"
    if calib_dir.exists():
        calib_images = sorted(calib_dir.glob("*.jpg"))[:5]
        image_paths.extend(calib_images)

    web_dir = REPO_ROOT / "datasets/web_raw"
    if web_dir.exists():
        image_paths.extend(web_dir.glob("*.jpg"))

    if not image_paths:
        print("[错误] 未找到图片")
        sys.exit(1)

    image_paths = sorted(list(set(image_paths)))

    model_path = REPO_ROOT / "artifacts/models/yolov8n_person_80map.onnx"
    batch_inference(image_paths, model_path, max_images=10)

    print("\n[完成] 批量推理完成！")
    print("结果保存在: artifacts/batch_result_*.jpg")
