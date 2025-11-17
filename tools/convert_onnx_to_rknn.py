#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
from importlib.metadata import version, PackageNotFoundError

# Expose RKNN at module scope so tests can monkeypatch tools.convert_onnx_to_rknn.RKNN
RKNN = None
try:
    from rknn.api import RKNN  # type: ignore
except Exception:
    # Keep RKNN as None if import fails or toolkit unavailable
    RKNN = None


def _detect_rknn_default_qdtype():
    try:
        ver = version('rknn-toolkit2')
        parts = ver.split('.')
        major = int(parts[0]) if parts and parts[0].isdigit() else 0
    except PackageNotFoundError:
        major = 0
    # rknn-toolkit2 >=2.x uses enums like 'w8a8'; 1.x uses 'asymmetric_quantized-u8'.
    return 'w8a8' if major >= 2 else 'asymmetric_quantized-u8'


def build_rknn(
    onnx_path: Path,
    out_path: Path,
    calib: Path = None,
    do_quant: bool = True,
    target: str = 'rk3588',
    quantized_dtype: str = None,
    mean: str = '0,0,0',
    std: str = '255,255,255',
    reorder: str = '2 1 0',  # BGR->RGB
):
    # Check if RKNN is available (module-level import may have failed)
    if RKNN is None:
        raise SystemExit(
            "rknn-toolkit2 not installed or not importable. Please run: pip install rknn-toolkit2"
        )

    mean_values = [[float(v) for v in mean.split(',')]]
    std_values = [[float(v) for v in std.split(',')]]

    onnx_path = Path(onnx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to instantiate RKNN (may fail if version incompatible or other issues)
    try:
        rknn = RKNN(verbose=True)
    except (ImportError, TypeError) as e:
        raise SystemExit(
            f"rknn-toolkit2 not installed. Please run: pip install rknn-toolkit2\nError: {e}"
        )
    except AttributeError as e:
        raise SystemExit(
            f"rknn-toolkit2 version incompatible. Please ensure rknn-toolkit2>=2.3.2 is installed.\nError: {e}"
        )
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

    print(f'Loading ONNX: {onnx_path}')
    try:
        ret = rknn.load_onnx(model=str(onnx_path))
    except Exception as e:
        # Translate protobuf/onnx parsing errors into ValueError for test determinism
        rknn.release()
        raise ValueError(f"Failed to load ONNX model: {e}") from e

    if ret != 0:
        print('load_onnx failed')
        rknn.release()
        sys.exit(1)

    dataset = None
    if do_quant:
        if calib is None:
            raise SystemExit('INT8 quantization requested but no calibration dataset provided')
        dataset = str(calib)
        if not Path(dataset).exists():
            raise SystemExit(f'Calibration file or folder not found: {dataset}')

    print('Building RKNN...')
    try:
        ret = rknn.build(do_quantization=bool(do_quant), dataset=dataset)
    except Exception as e:
        rknn.release()
        raise ValueError(f"Failed to build RKNN model: {e}") from e

    if ret != 0:
        print('build failed')
        rknn.release()
        sys.exit(1)

    print(f'Exporting RKNN to: {out_path}')
    try:
        ret = rknn.export_rknn(str(out_path))
    except Exception as e:
        rknn.release()
        raise ValueError(f"Failed to export RKNN model: {e}") from e

    if ret != 0:
        print('export_rknn failed')
        rknn.release()
        sys.exit(1)

    print('Done.')
    rknn.release()


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
            raise SystemExit(f'No images found in calibration folder: {calib}')
        list_path = calib / 'calib.txt'
        with open(list_path, 'w') as f:
            f.write('\n'.join(images))
        calib = list_path

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

if __name__ == '__main__':
    main()
