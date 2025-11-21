#!/usr/bin/env python3
"""ONNX to RKNN model conversion tool.

This script converts ONNX models to RKNN format with optional INT8 quantization
for deployment on Rockchip NPU platforms.
"""
import argparse
from pathlib import Path
import sys
from importlib.metadata import version, PackageNotFoundError
from contextlib import contextmanager
from typing import Optional

# Import custom exceptions
from apps.exceptions import ModelLoadError, ConfigurationError
from apps.logger import setup_logger

# Setup logger
logger = setup_logger(__name__, level='INFO')


@contextmanager
def rknn_context(verbose: bool = True):
    """Context manager for RKNN toolkit to ensure proper resource cleanup.

    Ensures rknn.release() is called even if exceptions occur, preventing
    GPU/memory leaks in error scenarios.

    Args:
        verbose: Enable verbose logging from RKNN toolkit

    Yields:
        RKNN: Initialized RKNN toolkit instance

    Example:
        >>> with rknn_context() as rknn:
        ...     rknn.load_onnx('model.onnx')
        ...     rknn.build(do_quantization=True, dataset='calib.txt')
        ...     rknn.export_rknn('model.rknn')
        ... # rknn.release() automatically called here
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

    rknn = RKNN(verbose=verbose)
    try:
        yield rknn
    finally:
        # Always release resources, even on exception
        try:
            rknn.release()
            logger.debug("RKNN resources released")
        except Exception as e:
            # Log but don't raise - we're in cleanup phase
            logger.warning(f"Failed to release RKNN resources (may already be released): {e}")


def _detect_rknn_default_qdtype():
    try:
        ver = version('rknn-toolkit2')
        parts = ver.split('.')
        major = int(parts[0]) if parts and parts[0].isdigit() else 0
    except PackageNotFoundError:
        major = 0
    # rknn-toolkit2 >=2.x uses enums like 'w8a8'; 1.x uses 'asymmetric_quantized-u8'.
    return 'w8a8' if major >= 2 else 'asymmetric_quantized-u8'


def _parse_and_validate_mean_std(mean: str, std: str):
    """Parse and validate mean/std parameters.

    Args:
        mean: Comma-separated mean values (e.g., '0,0,0')
        std: Comma-separated std values (e.g., '255,255,255')

    Returns:
        Tuple of (mean_values, std_values) as nested lists

    Raises:
        ValueError: If format is invalid or values are out of range
    """
    try:
        mean_list = [float(v.strip()) for v in mean.split(',')]
        std_list = [float(v.strip()) for v in std.split(',')]
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
        raise ValueError(
            f"Std must have exactly 3 values (R,G,B), got {len(std_list)}: {std_list}"
        )

    if any(s == 0 for s in std_list):
        raise ValueError(
            f"Std values cannot be zero (would cause division by zero): {std_list}"
        )

    if any(s < 0 for s in std_list):
        raise ValueError(
            f"Std values must be positive: {std_list}"
        )

    return [mean_list], [std_list]


def build_rknn(
    onnx_path: Path,
    out_path: Path,
    calib: Optional[Path] = None,
    do_quant: bool = True,
    target: str = 'rk3588',
    quantized_dtype: Optional[str] = None,
    mean: str = '0,0,0',
    std: str = '255,255,255',
    reorder: str = '2 1 0',  # BGR->RGB
):
    """Build RKNN model from ONNX with automatic resource management.

    Uses context manager to ensure RKNN resources are properly released
    even if exceptions occur during conversion.

    Args:
        onnx_path: Path to input ONNX model
        out_path: Path to output RKNN model
        calib: Optional calibration dataset path
        do_quant: Enable INT8 quantization
        target: Target platform (default: rk3588)
        quantized_dtype: Quantization dtype (auto-detected if None)
        mean: Mean values for normalization
        std: Std values for normalization
        reorder: Channel reordering (e.g., '2 1 0' for BGR->RGB)

    Raises:
        ModelLoadError: If model loading/building/exporting fails
        ConfigurationError: If configuration is invalid
    """
    # Validate and parse mean/std parameters
    mean_values, std_values = _parse_and_validate_mean_std(mean, std)

    onnx_path = Path(onnx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use context manager for automatic resource cleanup
    with rknn_context(verbose=True) as rknn:
        print('Configuring RKNN...')
        # Choose sensible default qdtype by toolkit major version if not provided
        if quantized_dtype in (None, ''):
            quantized_dtype = _detect_rknn_default_qdtype()
            print(f'Auto-select quantized_dtype={quantized_dtype}')

        rknn.config(
            target_platform=target,
            mean_values=mean_values,
            std_values=std_values,
            quantized_dtype=quantized_dtype,
        )

        logger.info(f'Loading ONNX: {onnx_path}')
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise ModelLoadError(f'Failed to load ONNX model: {onnx_path}')

        dataset = None
        if do_quant:
            if calib is None:
                raise ConfigurationError('INT8 quantization requested but no calibration dataset provided')
            dataset = str(calib)
            if not Path(dataset).exists():
                raise ConfigurationError(f'Calibration file or folder not found: {dataset}')

        logger.info('Building RKNN model...')
        ret = rknn.build(do_quantization=bool(do_quant), dataset=dataset)
        if ret != 0:
            raise ModelLoadError('Failed to build RKNN model')

        logger.info(f'Exporting RKNN to: {out_path}')
        ret = rknn.export_rknn(str(out_path))
        if ret != 0:
            raise ModelLoadError(f'Failed to export RKNN model to: {out_path}')

        logger.info('RKNN conversion completed successfully')
        # rknn.release() automatically called by context manager


def main():
    ap = argparse.ArgumentParser(description='Convert ONNX to RKNN (INT8 optional)')
    ap.add_argument('--onnx', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('artifacts/models/yolov8s_int8.rknn'))
    ap.add_argument('--calib', type=Path, help='calibration txt or image folder')
    ap.add_argument('--no-quant', dest='do_quant', action='store_false', default=True)
    ap.add_argument('--target', type=str, default='rk3588')
    ap.add_argument('--quant-dtype', type=str, default=None, help="For rknn-toolkit2>=2.x use 'w8a8'; for 1.x use 'asymmetric_quantized-u8'. If omitted, auto-detect.")
    ap.add_argument('--mean', type=str, default='0,0,0')
    ap.add_argument('--std', type=str, default='255,255,255')
    ap.add_argument('--reorder', type=str, default='2 1 0')
    args = ap.parse_args()

    calib = args.calib
    # If a folder is given for calib, create a temporary txt list
    if calib and calib.is_dir():
        from glob import glob
        images = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            images.extend(sorted(glob(str(calib / ext))))
        if not images:
            raise ConfigurationError(f'No images found in calibration folder: {calib}')
        list_path = calib / 'calib.txt'
        with open(list_path, 'w') as f:
            f.write('\n'.join(images))
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
    except (ModelLoadError, ConfigurationError) as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during conversion: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
