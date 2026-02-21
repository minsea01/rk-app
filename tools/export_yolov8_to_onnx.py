#!/usr/bin/env python3
"""YOLOv8/YOLO11 模型导出为 ONNX 格式。

整体流程：
    PyTorch 权重 (.pt)
        ↓  本脚本
    ONNX 模型 (.onnx)
        ↓  tools/convert_onnx_to_rknn.py
    RKNN 量化模型 (.rknn)
        ↓  部署到 RK3588 NPU

注意事项：
    - opset 必须用 12，RKNN toolkit2 对高版本算子支持有限
    - imgsz 建议用 416（生产环境），640 会导致 Transpose 算子超过
      NPU 16384 元素限制，退化为 CPU 执行
    - simplify=True 可消除冗余算子，提升后续 RKNN 转换成功率
"""

import argparse
from pathlib import Path
import sys

# 使用项目自定义异常，而非裸 Exception，便于上层统一处理错误类型
from apps.exceptions import ModelLoadError, ConfigurationError
from apps.logger import setup_logger

# 统一使用项目日志，不用 print
logger = setup_logger(__name__, level="INFO")


def export(
    weights: str,   # PyTorch 权重文件路径，如 yolo11n.pt
    imgsz: int,     # 输入图像尺寸（正方形边长），影响 ONNX 静态 shape
    opset: int,     # ONNX 算子集版本，RKNN 要求 ≤ 12
    simplify: bool, # 是否使用 onnx-simplifier 化简计算图
    dynamic: bool,  # 是否导出动态 batch size（RKNN 转换时须关闭）
    half: bool,     # 是否导出 FP16（仅 GPU 支持，RKNN 用 INT8 量化，通常不需要）
    outdir: Path,   # 输出目录，如 artifacts/models/
    outfile: str = None,  # 自定义输出文件名，默认沿用 Ultralytics 生成的名字
):
    # 确保输出目录存在，不存在则递归创建
    outdir.mkdir(parents=True, exist_ok=True)

    # 延迟导入：避免在没有安装 ultralytics 的环境（如仅跑 RKNN 推理的板端）报错
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ConfigurationError(
            f"Ultralytics not installed. Please run: pip install ultralytics\nError: {e}"
        ) from e

    # 检查权重文件是否存在，提前报错比等 YOLO() 内部崩溃信息更清晰
    weights_path = Path(weights)
    if not weights_path.exists():
        raise ModelLoadError(f"Weights file not found: {weights}")

    logger.info(f"Loading model: {weights}")
    try:
        # 用 Ultralytics 官方接口加载 .pt 权重，内部会自动识别模型结构
        model = YOLO(weights)
    except (RuntimeError, ValueError, FileNotFoundError, Exception) as e:
        raise ModelLoadError(f"Failed to load model from {weights}: {e}") from e

    logger.info(f"Exporting to ONNX (imgsz={imgsz}, opset={opset}, simplify={simplify})")
    try:
        # 核心导出调用：Ultralytics 内部调用 torch.onnx.export，
        # 然后可选地调用 onnxsim.simplify 化简计算图
        onnx_path = model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,      # RKNN toolkit2 推荐 opset=12
            simplify=simplify, # 化简后算子更少，RKNN 转换更稳定
            dynamic=dynamic,  # RKNN 转换要求静态 shape，保持 False
            half=half,        # FP16 导出，RKNN INT8 量化流程不需要
        )
    except (RuntimeError, ValueError, TypeError) as e:
        raise ModelLoadError(f"Failed to export model to ONNX: {e}") from e

    # Ultralytics 默认把 ONNX 写到与 .pt 同目录，需要搬到我们指定的 outdir
    onnx_path = Path(onnx_path)
    target = outdir / (outfile if outfile else onnx_path.name)
    if onnx_path.resolve() != target.resolve():
        try:
            # 用字节复制而非 shutil.move，避免跨设备移动失败
            target.write_bytes(onnx_path.read_bytes())
        except (IOError, OSError, PermissionError) as e:
            raise ModelLoadError(f"Failed to move ONNX file to {target}: {e}") from e

    logger.info(f"Successfully exported ONNX: {target}")
    return target


def main():
    p = argparse.ArgumentParser(description="Export YOLOv8/YOLO11 .pt -> .onnx (for RKNN conversion)")
    # 输入权重文件，支持 yolov8n.pt / yolo11n.pt 等所有 Ultralytics 模型
    p.add_argument("--weights", type=str, default="yolov8s.pt", help="YOLOv8 .pt weights")
    # 图像尺寸：生产用 416（避免 NPU Transpose 限制），开发验证可用 640
    p.add_argument("--imgsz", type=int, default=640)
    # ONNX opset 版本：RKNN toolkit2 要求 ≤ 12
    p.add_argument("--opset", type=int, default=12)
    # 是否化简计算图（推荐开启）
    p.add_argument("--simplify", action="store_true", default=True)
    p.add_argument("--no-simplify", dest="simplify", action="store_false")
    # dynamic=True 时 batch 维度为动态，RKNN 转换不支持，保持 False
    p.add_argument("--dynamic", action="store_true", default=False)
    # half=True 导出 FP16，RKNN 走 INT8 量化，一般不需要
    p.add_argument("--half", action="store_true", default=False)
    # ONNX 输出目录
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
