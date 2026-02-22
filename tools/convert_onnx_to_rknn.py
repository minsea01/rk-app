#!/usr/bin/env python3
"""ONNX 模型转换为 RKNN 格式。

整体流程：
    ONNX 模型 (.onnx)
        ↓  本脚本（rknn-toolkit2 PC 端）
    RKNN 量化模型 (.rknn)
        ↓  部署到 RK3588 NPU
    板端用 rknn-toolkit2-lite 加载推理

INT8 量化原理（为什么需要 calibration dataset）：
    ONNX 是 FP32 权重，NPU 只支持 INT8 整数运算。
    量化 = 把 FP32 值映射到 [-128, 127]，需要知道每一层激活值的实际分布范围。
    calibration dataset 就是用来"观测"这个范围的，图片越有代表性，精度损失越小。
    建议用 100~300 张与实际场景相似的图片。

注意事项：
    - calibration 路径必须是绝对路径，否则 RKNN toolkit 静默失败
    - mean/std 必须与训练时的预处理一致（YOLO 默认 mean=0,0,0  std=255,255,255）
    - reorder='2 1 0' 表示 BGR→RGB 通道转换（OpenCV 读图是 BGR，模型训练用 RGB）
    - rknn-toolkit2 ≥2.x 量化 dtype 用 'w8a8'；1.x 用 'asymmetric_quantized-u8'
"""

import argparse
import logging
from pathlib import Path
import sys
from importlib.metadata import version, PackageNotFoundError
from contextlib import contextmanager
from typing import Optional

# 导入项目内自定义异常
from apps.exceptions import ModelLoadError, ConfigurationError
from apps.logger import setup_logger

# 初始化日志器
logger = setup_logger(__name__, level=logging.INFO)


@contextmanager
def rknn_context(verbose: bool = True):
    """RKNN 工具链上下文管理器，确保资源被正确释放。

    即使转换过程中发生异常，也会在 finally 中调用 `rknn.release()`，
    避免 GPU/内存资源泄漏。

    参数:
        verbose: 是否开启 RKNN toolkit 的详细日志输出

    产出:
        RKNN: 已初始化的 RKNN toolkit 实例

    示例:
        >>> with rknn_context() as rknn:
        ...     rknn.load_onnx('model.onnx')
        ...     rknn.build(do_quantization=True, dataset='calib.txt')
        ...     rknn.export_rknn('model.rknn')
        ... # 退出 with 后会自动调用 rknn.release()
    """
    try:
        from rknn.api import RKNN
    except ImportError as e:
        raise ConfigurationError(
            f"rknn-toolkit2 not installed. Please run: pip install rknn-toolkit2\nError: {e}"
        ) from e
    except (AttributeError, TypeError) as e:
        raise ConfigurationError(
            f"rknn-toolkit2 version incompatible. Please ensure rknn-toolkit2>=2.3.2 is installed.\nError: {e}"
        ) from e

    try:
        rknn = RKNN(verbose=verbose)
    except (TypeError, AttributeError) as e:
        # TypeError: 不同 SDK 版本构造函数签名变化
        # AttributeError: RKNN 类缺少预期属性
        raise ConfigurationError(
            f"rknn-toolkit2 version incompatible. Please ensure rknn-toolkit2>=2.3.2 is installed.\nError: {e}"
        ) from e

    try:
        yield rknn
    finally:
        # 无论是否异常都尝试释放资源
        try:
            rknn.release()
            logger.debug("RKNN resources released")
        except (RuntimeError, AttributeError, OSError) as e:
            # 清理阶段仅记录日志，不再抛出异常
            # RuntimeError: RKNN SDK 内部错误
            # AttributeError: RKNN 对象已释放或无效
            # OSError: GPU/内存清理失败
            logger.warning(f"Failed to release RKNN resources (may already be released): {e}")


def _detect_rknn_default_qdtype():
    try:
        ver = version("rknn-toolkit2")
        parts = ver.split(".")
        major = int(parts[0]) if parts and parts[0].isdigit() else 0
    except PackageNotFoundError:
        major = 0
    # rknn-toolkit2 >=2.x 使用 'w8a8'；1.x 使用 'asymmetric_quantized-u8'
    return "w8a8" if major >= 2 else "asymmetric_quantized-u8"


def _parse_and_validate_mean_std(mean: str, std: str):
    """解析并校验 mean/std 参数。

    参数:
        mean: 逗号分隔的均值，例如 '0,0,0'
        std: 逗号分隔的标准差，例如 '255,255,255'

    返回:
        (mean_values, std_values) 元组，均为嵌套列表格式

    异常:
        ValueError: 格式非法或取值不合法时抛出
    """
    try:
        mean_list = [float(v.strip()) for v in mean.split(",")]
        std_list = [float(v.strip()) for v in std.split(",")]
    except ValueError as e:
        raise ValueError(
            f"Invalid mean/std format. Expected comma-separated floats.\n"
            f"  mean='{mean}', std='{std}'\n"
            f"  Error: {e}"
        )

    if len(mean_list) != 3:
        raise ValueError(
            f"Mean must have exactly 3 values (R,G,B), got {len(mean_list)}: {mean_list}"
        )

    if len(std_list) != 3:
        raise ValueError(f"Std must have exactly 3 values (R,G,B), got {len(std_list)}: {std_list}")

    if any(s == 0 for s in std_list):
        raise ValueError(f"Std values cannot be zero (would cause division by zero): {std_list}")

    if any(s < 0 for s in std_list):
        raise ValueError(f"Std values must be positive: {std_list}")

    return [mean_list], [std_list]


def build_rknn(
    onnx_path: Path,
    out_path: Path,
    calib: Optional[Path] = None,
    do_quant: bool = True,
    target: str = "rk3588",
    quantized_dtype: Optional[str] = None,
    mean: str = "0,0,0",
    std: str = "255,255,255",
    reorder: str = "2 1 0",  # BGR->RGB
):
    """将 ONNX 模型转换为 RKNN，并自动管理资源释放。

    通过上下文管理器保证在转换异常时也能正确释放 RKNN 资源。

    参数:
        onnx_path: 输入 ONNX 模型路径
        out_path: 输出 RKNN 模型路径
        calib: 可选的校准数据集路径
        do_quant: 是否启用 INT8 量化
        target: 目标平台（默认 rk3588）
        quantized_dtype: 量化类型（为 None 时自动检测）
        mean: 归一化均值
        std: 归一化标准差
        reorder: 通道重排（如 '2 1 0' 表示 BGR->RGB）

    异常:
        ModelLoadError: 模型加载/构建/导出失败
        ConfigurationError: 配置参数不合法
    """
    # 校验并解析 mean/std 参数
    mean_values, std_values = _parse_and_validate_mean_std(mean, std)

    onnx_path = Path(onnx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 使用上下文管理器自动清理 RKNN 资源
    with rknn_context(verbose=True) as rknn:
        print("Configuring RKNN...")
        # 若未显式传入 quantized_dtype，则根据 toolkit 主版本自动选择默认值
        if quantized_dtype in (None, ""):
            quantized_dtype = _detect_rknn_default_qdtype()
            print(f"Auto-select quantized_dtype={quantized_dtype}")

        config_kwargs = {
            "target_platform": target,
            "mean_values": mean_values,
            "std_values": std_values,
            "quantized_dtype": quantized_dtype,
        }
        if reorder:
            config_kwargs["reorder_channel"] = reorder

        try:
            rknn.config(**config_kwargs)
        except TypeError as e:
            # 某些旧版或变体 toolkit 可能不支持 reorder_channel 参数
            if "reorder_channel" in config_kwargs and "reorder_channel" in str(e):
                logger.warning(
                    "Current RKNN toolkit does not support reorder_channel, "
                    "falling back without explicit channel reorder"
                )
                config_kwargs.pop("reorder_channel", None)
                rknn.config(**config_kwargs)
            else:
                raise

        logger.info(f"Loading ONNX: {onnx_path}")
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise ModelLoadError(f"Failed to load ONNX model: {onnx_path}")

        dataset = None
        if do_quant:
            if calib is None:
                raise ConfigurationError(
                    "INT8 quantization requested but no calibration dataset provided"
                )
            dataset = str(calib)
            if not Path(dataset).exists():
                raise ConfigurationError(f"Calibration file or folder not found: {dataset}")

        logger.info("Building RKNN model...")
        ret = rknn.build(do_quantization=bool(do_quant), dataset=dataset)
        if ret != 0:
            raise ModelLoadError("Failed to build RKNN model")

        logger.info(f"Exporting RKNN to: {out_path}")
        ret = rknn.export_rknn(str(out_path))
        if ret != 0:
            raise ModelLoadError(f"Failed to export RKNN model to: {out_path}")

        logger.info("RKNN conversion completed successfully")
        # rknn.release() 会在上下文退出时自动调用


def main():
    ap = argparse.ArgumentParser(description="Convert ONNX to RKNN (INT8 optional)")
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/models/yolov8s_int8.rknn"))
    ap.add_argument("--calib", type=Path, help="calibration txt or image folder")
    ap.add_argument("--no-quant", dest="do_quant", action="store_false", default=True)
    ap.add_argument("--target", type=str, default="rk3588")
    ap.add_argument(
        "--quant-dtype",
        type=str,
        default=None,
        help="For rknn-toolkit2>=2.x use 'w8a8'; for 1.x use 'asymmetric_quantized-u8'. If omitted, auto-detect.",
    )
    ap.add_argument("--mean", type=str, default="0,0,0")
    ap.add_argument("--std", type=str, default="255,255,255")
    ap.add_argument("--reorder", type=str, default="2 1 0")
    args = ap.parse_args()

    calib = args.calib
    # 如果 calib 传入的是目录，则自动生成临时 calib.txt（每行一个图片路径）
    if calib and calib.is_dir():
        from glob import glob

        images = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            images.extend(sorted(str(Path(p).resolve()) for p in glob(str(calib / ext))))
        if not images:
            raise ConfigurationError(f"No images found in calibration folder: {calib}")
        list_path = calib / "calib.txt"
        with open(list_path, "w") as f:
            f.write("\n".join(images))
        calib = list_path

    try:
        build_rknn(
            onnx_path=args.onnx,
            out_path=args.out,
            calib=calib,
            do_quant=args.do_quant,
            target=args.target,
            quantized_dtype=args.quant_dtype,
            mean=args.mean,
            std=args.std,
            reorder=args.reorder,
        )
        return 0
    except (ModelLoadError, ConfigurationError, ValueError) as e:
        logger.error(f"Conversion failed: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
