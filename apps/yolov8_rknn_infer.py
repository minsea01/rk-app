#!/usr/bin/env python3
"""RK3588 板端 RKNNLite 推理示例（面向初学者）。

本文件负责：
1. 解析命令行参数（模型路径、阈值、预处理选项等）
2. 加载 RKNN 模型并初始化 NPU 运行时
3. 读取单张图片或摄像头视频流
4. 执行预处理 -> 推理 -> 解码 -> 可视化

建议阅读顺序：
`build_arg_parser()` -> `main()` -> 图像分支 -> 摄像头分支。
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from apps.deprecation import warn_deprecated
from apps.utils.decode import decode_predictions as _decode_predictions
from apps.utils.decode_meta import load_decode_meta
from apps.utils.preprocess_pipeline import (
    PreprocessState,
    build_preprocess_config,
    map_boxes_back,
    run_preprocess,
)
from apps.utils.headless import safe_imshow, safe_waitKey
from apps.exceptions import RKNNError, PreprocessError, InferenceError, ModelLoadError
from apps.logger import setup_logger


def decode_predictions(
    pred,
    imgsz,
    conf_thres,
    iou_thres,
    head="auto",
    ratio_pad=(1.0, (0.0, 0.0)),
    orig_shape=None,
    decode_meta=None,
):
    """兼容层：旧调用入口，内部转发到新解码函数。

    这里保留该函数是为了兼容旧代码，真正实现位于
    `apps.utils.decode.decode_predictions`。
    """
    warn_deprecated(
        "apps.yolov8_rknn_infer.decode_predictions",
        "apps.utils.decode.decode_predictions",
        once=True,
    )
    return _decode_predictions(
        pred=pred,
        imgsz=imgsz,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        head=head,
        ratio_pad=ratio_pad,
        orig_shape=orig_shape,
        decode_meta=decode_meta,
    )


def load_labels(names_path: Path):
    """从 names.txt 读取类别名列表。

    返回:
        list[str] | None:
        - 文件存在时返回类别名列表
        - 未提供路径或文件不存在时返回 None（回退为显示类别 id）
    """
    if names_path and names_path.exists():
        return [x.strip() for x in names_path.read_text().splitlines() if x.strip()]
    return None


def draw_boxes(img, boxes, confs, class_ids, names=None):
    """在图像上绘制检测框和标签。

    约定:
    - `boxes` 形状一般为 [N, 4]，坐标格式 x1,y1,x2,y2
    - `confs` 和 `class_ids` 长度需与 `boxes` 一致
    """
    for (x1, y1, x2, y2), c, cls in zip(boxes.astype(int), confs, class_ids):
        label = f"{int(cls)}:{c:.2f}"
        if names and 0 <= int(cls) < len(names):
            label = f"{names[int(cls)]}:{c:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img


def build_arg_parser() -> argparse.ArgumentParser:
    """定义命令行参数。

    注意：
    - 本脚本优先面向板端部署，因此默认 `--imgsz 416` 与 `--core-mask 0x7`
    - 大部分预处理参数可通过 `--cfg` 和 CLI 覆盖合并
    """
    ap = argparse.ArgumentParser(description="YOLOv8 RKNNLite inference on RK3588")
    ap.set_defaults(
        undistort_enable=None,
        roi_enable=None,
        gamma_enable=None,
        white_balance_enable=None,
        denoise_enable=None,
        roi_clamp=None,
    )
    ap.add_argument("--model", type=Path, required=True, help="path to .rknn model file")
    ap.add_argument("--source", type=Path, help="image path; if omitted, opens /dev/video0")
    ap.add_argument("--cfg", type=Path, default=None, help="optional YAML config file")
    # 默认使用 416 分辨率以避免 Transpose 回退并提升 NPU 吞吐
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="confidence threshold (default: 0.5, recommended >=0.5 for production to avoid NMS bottleneck)",
    )
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--names", type=Path, default=None, help="names.txt (one class per line)")
    ap.add_argument(
        "--head",
        type=str,
        choices=["auto", "dfl", "raw"],
        default="auto",
        help="head decode type (auto uses decode metadata when available)",
    )
    ap.add_argument(
        "--pp-profile", type=str, default=None, choices=["speed", "balanced", "quality"]
    )
    ap.add_argument("--undistort-enable", action="store_true")
    ap.add_argument("--undistort-calib", type=str, default=None)
    ap.add_argument("--roi-enable", action="store_true")
    ap.add_argument("--roi-mode", type=str, default=None, choices=["normalized", "pixel"])
    ap.add_argument("--roi-norm", type=str, default=None, help="normalized ROI x,y,w,h in [0,1]")
    ap.add_argument("--roi-px", type=str, default=None, help="pixel ROI x,y,w,h")
    ap.add_argument("--roi-min-size", type=int, default=None)
    ap.add_argument("--roi-no-clamp", dest="roi_clamp", action="store_false")
    ap.add_argument("--gamma-enable", action="store_true")
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--white-balance-enable", action="store_true")
    ap.add_argument("--white-balance-clip", type=float, default=None)
    ap.add_argument("--denoise-enable", action="store_true")
    ap.add_argument("--denoise-method", type=str, default=None, choices=["bilateral"])
    ap.add_argument("--denoise-d", type=int, default=None)
    ap.add_argument("--denoise-sigma-color", type=float, default=None)
    ap.add_argument("--denoise-sigma-space", type=float, default=None)
    ap.add_argument(
        "--input-format",
        type=str,
        default=None,
        choices=["auto", "bgr", "rgb", "gray", "bayer_rg", "bayer_bg", "bayer_gr", "bayer_gb"],
    )
    # 默认三核全开，满足“轻量化+多核并行”要求
    ap.add_argument(
        "--core-mask",
        type=lambda x: int(x, 0),
        default=0x7,
        help="NPU core mask (e.g., 0x7 for 3 cores)",
    )
    ap.add_argument("--save", type=Path, default=None, help="save annotated result image")
    ap.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="logging verbosity",
    )
    return ap


def main():
    """主入口：完成参数读取、模型初始化、推理循环与资源清理。"""
    ap = build_arg_parser()
    args = ap.parse_args()

    logger = setup_logger(__name__, level=getattr(logging, args.log_level.upper()))
    # 将命令行覆盖项整理成字典，后续交给预处理配置构建器做“默认值 + 文件 + CLI”合并。
    preprocess_overrides = {
        "profile": args.pp_profile,
        "undistort_enable": args.undistort_enable,
        "calibration_file": args.undistort_calib,
        "roi_enable": args.roi_enable,
        "roi_mode": args.roi_mode,
        "roi_normalized_xywh": args.roi_norm,
        "roi_pixel_xywh": args.roi_px,
        "roi_min_size": args.roi_min_size,
        "roi_clamp": args.roi_clamp,
        "gamma_enable": args.gamma_enable,
        "gamma_value": args.gamma,
        "white_balance_enable": args.white_balance_enable,
        "white_balance_clip_percent": args.white_balance_clip,
        "denoise_enable": args.denoise_enable,
        "denoise_method": args.denoise_method,
        "denoise_d": args.denoise_d,
        "denoise_sigma_color": args.denoise_sigma_color,
        "denoise_sigma_space": args.denoise_sigma_space,
        "input_format": args.input_format,
    }
    try:
        preprocess_config = build_preprocess_config(args.cfg, preprocess_overrides, logger=logger)
    except (ValueError, OSError) as e:
        raise PreprocessError(f"Failed to load preprocess config: {e}") from e
    # 预处理状态（例如某些缓存/上下文）统一复用，避免每帧重复初始化。
    preprocess_state = PreprocessState()

    # `--source` 传入时按“单图模式”处理；不传时走摄像头模式。
    if args.source is not None:
        if not args.source.exists():
            raise PreprocessError(f"Source path not found: {args.source}")
        if not args.source.is_file():
            raise PreprocessError(
                "Source must be an image file. Omit --source to use the camera: " f"{args.source}"
            )

    try:
        from rknnlite.api import RKNNLite
    except ImportError as e:
        raise SystemExit(
            f"rknn-toolkit-lite2 not installed on device.\n"
            f"Install with: pip install rknn-toolkit-lite2\n"
            f"Error: {e}"
        )

    rknn = RKNNLite()

    # 1) 加载 RKNN 模型文件（仅加载，不执行硬件初始化）
    logger.info("Loading RKNN: %s", args.model)
    if not args.model.exists():
        raise ModelLoadError(f"Model file not found: {args.model}")

    try:
        ret = rknn.load_rknn(str(args.model))
        if ret != 0:
            raise ModelLoadError(f"Failed to load RKNN model: {args.model}")
    except ModelLoadError:
        raise
    except (IOError, OSError) as e:
        raise ModelLoadError(f"Error reading RKNN model file: {e}") from e

    # 2) 初始化 RKNN 运行时（会与 NPU 交互，core_mask 决定启用哪些核心）
    logger.info("Initializing runtime, core_mask=%s", hex(args.core_mask))
    try:
        ret = rknn.init_runtime(core_mask=args.core_mask)
        if ret != 0:
            raise RKNNError(
                f"Failed to initialize RKNN runtime with core_mask={hex(args.core_mask)}"
            )
    except RKNNError:
        raise
    except RuntimeError as e:
        raise RKNNError(f"RKNN runtime initialization error: {e}") from e

    # 类别名和解码元信息都属于“推理后处理配置”，只需加载一次。
    class_names = load_labels(args.names)
    decode_meta = load_decode_meta(args.model, logger=logger)

    # ------------------------ 单张图片推理路径 ------------------------
    if args.source is not None:
        # 3) 读取并预处理图片
        try:
            img0 = cv2.imread(str(args.source))
            if img0 is None:
                raise PreprocessError(f"Failed to read image: {args.source}")
            img, frame_meta, coord_frame = run_preprocess(
                img0, args.imgsz, preprocess_config, preprocess_state, logger=logger
            )
        except PreprocessError:
            raise
        except (cv2.error, ValueError) as e:
            raise PreprocessError(f"Error preprocessing image: {e}") from e

        # 4) 执行推理
        # RKNN 转换时通常已配置 BGR->RGB 与 /255，因此这里直接喂 BGR uint8 即可。
        # RKNNLite 接口要求带 batch 维度，因此把 (H, W, 3) 扩成 (1, H, W, 3)。
        input_data = np.expand_dims(img, axis=0)
        t0 = time.time()
        try:
            outputs = rknn.inference(inputs=[input_data])
        except (RuntimeError, ValueError) as e:
            raise InferenceError(f"RKNN inference failed: {e}") from e
        dt = (time.time() - t0) * 1000
        logger.info("Inference time: %.2f ms", dt)
        pred = outputs[0]
        # 某些模型输出可能是 [N, C]，补成 [1, N, C] 以统一后续 decode 逻辑。
        if pred.ndim == 2:
            pred = pred[None, ...]
        # 5) 解码 + NMS
        boxes, confs, cls_ids = decode_predictions(
            pred,
            args.imgsz,
            args.conf,
            args.iou,
            args.head,
            (1.0, (0.0, 0.0)),
            img.shape[:2],
            decode_meta=decode_meta,
        )
        # 将坐标从预处理后的坐标系映射回原图坐标系（例如 letterbox 逆变换）。
        boxes = map_boxes_back(boxes, frame_meta)
        logger.info("Detections: %d", len(boxes))
        vis = draw_boxes(coord_frame.copy(), boxes, confs, cls_ids, class_names)
        if args.save:
            cv2.imwrite(str(args.save), vis)
            logger.info("Saved: %s", args.save)
        else:
            # 自动处理无显示环境：headless 模式下会回退为写文件而不是弹窗。
            safe_imshow("result", vis, fallback_path="artifacts/result.jpg", wait_key=0)
        rknn.release()
        return

    # ------------------------ 摄像头流推理路径 ------------------------
    cap = None
    # 指数退避重连参数：摄像头暂不可用时避免疯狂重试。
    reconnect_delay = 0.5
    reconnect_max = 5.0
    reconnect_attempts = 0

    def open_camera():
        """尝试打开默认摄像头，失败则返回 None。"""
        cap_obj = cv2.VideoCapture(0)
        if not cap_obj.isOpened():
            cap_obj.release()
            return None
        return cap_obj

    try:
        cap = open_camera()
        if cap is None:
            # 注意：某些 OpenCV 后端即使打开失败也会返回对象，因此这里仍做防御式释放。
            if cap is not None:
                cap.release()
                cap = None  # Prevent double-release in finally block
            raise PreprocessError("Failed to open camera (/dev/video0)")

        # 记录每帧 FPS，最后输出平均值和 P90，便于快速做性能体感评估。
        fps_hist = []
        while True:
            if cap is None or not cap.isOpened():
                cap = open_camera()
                if cap is None:
                    reconnect_attempts += 1
                    if reconnect_attempts % 10 == 1:
                        logger.warning(
                            "Camera unavailable; retrying (attempt %d)", reconnect_attempts
                        )
                    time.sleep(reconnect_delay)
                    # 指数退避：0.5s -> 1s -> 2s -> ... -> 5s（上限）
                    reconnect_delay = min(reconnect_max, reconnect_delay * 2)
                    continue
                reconnect_attempts = 0
                reconnect_delay = 0.5

            ret, img0 = cap.read()
            if not ret:
                reconnect_attempts += 1
                if reconnect_attempts % 10 == 1:
                    logger.warning(
                        "Video capture failed; reconnecting (attempt %d)", reconnect_attempts
                    )
                cap.release()
                cap = None
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_max, reconnect_delay * 2)
                continue
            try:
                img, frame_meta, coord_frame = run_preprocess(
                    img0, args.imgsz, preprocess_config, preprocess_state, logger=logger
                )
                # Add batch dimension: (H, W, 3) -> (1, H, W, 3) to match image path
                input_data = np.expand_dims(img, axis=0)
                t0 = time.time()
                outputs = rknn.inference(inputs=[input_data])
                pred = outputs[0]
            except (cv2.error, RuntimeError, ValueError) as e:
                # 流媒体模式下单帧错误不应让整个进程退出，记录日志后跳过该帧。
                logger.warning("Inference error (skipping frame): %s", e)
                continue
            if pred.ndim == 2:
                pred = pred[None, ...]
            boxes, confs, cls_ids = decode_predictions(
                pred,
                args.imgsz,
                args.conf,
                args.iou,
                args.head,
                (1.0, (0.0, 0.0)),
                img.shape[:2],
                decode_meta=decode_meta,
            )
            boxes = map_boxes_back(boxes, frame_meta)
            t1 = time.time()
            fps = 1.0 / max(1e-6, (t1 - t0))
            fps_hist.append(fps)
            vis = draw_boxes(coord_frame.copy(), boxes, confs, cls_ids, class_names)
            cv2.putText(
                vis, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
            )

            # Auto-handles headless mode (saves to file instead of displaying)
            safe_imshow("result", vis, fallback_path="artifacts/camera_frame.jpg")

            # GUI 模式按 ESC 退出；headless 模式下 safe_waitKey 通常返回 -1。
            if safe_waitKey(1) & 0xFF == 27:
                break
    except PreprocessError:
        # Re-raise preprocessing errors
        raise
    except (cv2.error, RuntimeError) as e:
        # Wrap known errors that can occur during video processing
        raise PreprocessError(f"Camera processing error: {e}") from e
    finally:
        # 统一资源清理，确保异常路径也能释放摄像头/NPU/窗口。
        if cap is not None:
            cap.release()
        rknn.release()
        cv2.destroyAllWindows()
        if fps_hist:
            logger.info("Avg FPS: %.2f, P90: %.2f", np.mean(fps_hist), np.percentile(fps_hist, 90))


if __name__ == "__main__":
    main()
