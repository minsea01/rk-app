#!/usr/bin/env python3
"""
演示脚本：使用YOLO11n进行80类物体检测

这个脚本展示如何满足毕业设计任务书要求：
"检测或识别物体种类大于10种"

使用YOLO11n预训练模型，支持COCO 80类物体检测：
person, car, dog, cat, chair, bottle, etc.

用法：
    python scripts/demo_80classes.py --image assets/test.jpg
    python scripts/demo_80classes.py --image assets/test.jpg --output artifacts/result_80classes.jpg
"""
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from apps.logger import setup_logger
from apps.exceptions import ModelLoadError, PreprocessError

logger = setup_logger(__name__, level='INFO')


def load_class_names(names_file: Path):
    """加载COCO 80类名称列表"""
    if not names_file.exists():
        raise FileNotFoundError(f"Class names file not found: {names_file}")

    with open(names_file, 'r') as f:
        names = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(names)} class names from {names_file}")
    return names


def run_onnx_inference(model_path: Path, image_path: Path, class_names: list,
                       conf_threshold: float = 0.25, output_path: Path = None):
    """
    使用ONNX Runtime运行80类物体检测

    Args:
        model_path: ONNX模型路径
        image_path: 输入图像路径
        class_names: 80类名称列表
        conf_threshold: 置信度阈值
        output_path: 输出图像路径（可选）
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ModelLoadError("onnxruntime not installed. Run: pip install onnxruntime")

    # 检查模型文件
    if not model_path.exists():
        raise ModelLoadError(f"Model file not found: {model_path}")

    # 检查图像文件
    if not image_path.exists():
        raise PreprocessError(f"Image file not found: {image_path}")

    logger.info(f"Loading model: {model_path}")
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])

    # 获取输入输出信息
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    logger.info(f"Model input: {input_name}, shape: {input_shape}")

    # 读取并预处理图像
    logger.info(f"Loading image: {image_path}")
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise PreprocessError(f"Failed to read image: {image_path}")

    img_h, img_w = img_bgr.shape[:2]
    logger.info(f"Image size: {img_w}x{img_h}")

    # 预处理：resize + normalize
    target_size = 640  # YOLO输入尺寸
    img_resized = cv2.resize(img_bgr, (target_size, target_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_normalized, (2, 0, 1))[np.newaxis, ...]  # NCHW

    # 运行推理
    logger.info("Running inference...")
    outputs = session.run(None, {input_name: img_input})
    predictions = outputs[0]  # (1, 84, 8400) for YOLO11n

    logger.info(f"Predictions shape: {predictions.shape}")

    # 后处理：解析检测结果
    detections = []

    # YOLO11n输出格式: (1, 84, 8400)
    # 84 = 4(bbox) + 80(classes)
    if predictions.shape[1] == 84:
        pred = predictions[0].T  # (8400, 84)

        # 提取bbox和类别
        boxes = pred[:, :4]  # (8400, 4) - cx, cy, w, h
        class_scores = pred[:, 4:]  # (8400, 80)

        # 找到每个预测的最高分类和分数
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        # 过滤低置信度
        mask = confidences > conf_threshold

        valid_boxes = boxes[mask]
        valid_confs = confidences[mask]
        valid_class_ids = class_ids[mask]

        logger.info(f"Found {len(valid_boxes)} detections above threshold {conf_threshold}")

        # 转换bbox格式：cx,cy,w,h -> x1,y1,x2,y2
        for box, conf, cls_id in zip(valid_boxes, valid_confs, valid_class_ids):
            cx, cy, w, h = box
            x1 = (cx - w / 2) * img_w / target_size
            y1 = (cy - h / 2) * img_h / target_size
            x2 = (cx + w / 2) * img_w / target_size
            y2 = (cy + h / 2) * img_h / target_size

            detections.append({
                'class_id': int(cls_id),
                'class_name': class_names[int(cls_id)] if int(cls_id) < len(class_names) else f'class_{cls_id}',
                'confidence': float(conf),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })

    # 打印检测结果
    logger.info(f"\n{'='*60}")
    logger.info(f"Detection Results ({len(detections)} objects found):")
    logger.info(f"{'='*60}")

    detected_classes = set()
    for i, det in enumerate(detections, 1):
        logger.info(f"{i}. {det['class_name']}: {det['confidence']:.2%} confidence")
        detected_classes.add(det['class_name'])

    logger.info(f"{'='*60}")
    logger.info(f"Total unique classes detected: {len(detected_classes)}")
    logger.info(f"Classes: {', '.join(sorted(detected_classes))}")
    logger.info(f"{'='*60}\n")

    # 绘制检测框
    if output_path:
        img_result = img_bgr.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            class_name = det['class_name']
            conf = det['confidence']

            # 绘制边界框
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f"{class_name}: {conf:.2%}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_result, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(img_result, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imwrite(str(output_path), img_result)
        logger.info(f"Result saved to: {output_path}")

    return detections, detected_classes


def main():
    parser = argparse.ArgumentParser(
        description='YOLO11n 80类物体检测演示（满足毕业设计要求：检测物体种类≥10种）'
    )
    parser.add_argument('--model', type=Path,
                       default=Path('artifacts/models/yolo11n.onnx'),
                       help='ONNX model path (default: yolo11n.onnx)')
    parser.add_argument('--image', type=Path, required=True,
                       help='Input image path')
    parser.add_argument('--names', type=Path,
                       default=Path('config/coco80_names.txt'),
                       help='Class names file (default: coco80_names.txt)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output image path (optional)')

    args = parser.parse_args()

    # 加载类别名称
    class_names = load_class_names(args.names)

    logger.info(f"\n{'='*60}")
    logger.info("YOLO11n 80-Class Object Detection Demo")
    logger.info(f"{'='*60}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Image: {args.image}")
    logger.info(f"Classes: {len(class_names)} (COCO dataset)")
    logger.info(f"Confidence threshold: {args.conf}")
    logger.info(f"{'='*60}\n")

    # 运行推理
    detections, detected_classes = run_onnx_inference(
        args.model, args.image, class_names, args.conf, args.output
    )

    # 任务书要求验证
    required_classes = 10
    if len(detected_classes) >= required_classes:
        logger.info(f"✅ 满足任务书要求：检测物体种类 {len(detected_classes)} ≥ {required_classes}")
    else:
        logger.warning(f"⚠️ 当前图像仅检测到 {len(detected_classes)} 类物体")
        logger.info(f"   建议：使用包含更多物体类别的测试图像")
        logger.info(f"   模型支持80类检测，可检测：{', '.join(class_names[:10])}...")

    return 0


if __name__ == '__main__':
    sys.exit(main())
