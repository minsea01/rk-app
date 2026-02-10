#!/usr/bin/env python3
"""Deprecated wrapper for tools.convert_onnx_to_rknn."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.deprecation import warn_deprecated
from apps.exceptions import ConfigurationError, ModelLoadError
from tools.convert_onnx_to_rknn import build_rknn


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deprecated: use tools/convert_onnx_to_rknn.py --onnx ... --out ..."
    )
    parser.add_argument("onnx_model", type=Path, help="Path to ONNX model file")
    parser.add_argument("-d", "--dataset", type=Path, default=None, help="Calibration dataset (file or dir)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output RKNN model path")
    parser.add_argument("--target", type=str, default="rk3588", help="Target platform")
    return parser


def _default_output(onnx_path: Path, quantized: bool) -> Path:
    suffix = "_int8.rknn" if quantized else "_fp16.rknn"
    return onnx_path.with_name(f"{onnx_path.stem}{suffix}")


def main() -> int:
    args = build_arg_parser().parse_args()
    warn_deprecated("tools/export_rknn.py", "tools/convert_onnx_to_rknn.py", once=True)

    do_quant = args.dataset is not None
    out_path = args.output or _default_output(args.onnx_model, do_quant)

    try:
        build_rknn(
            onnx_path=args.onnx_model,
            out_path=out_path,
            calib=args.dataset,
            do_quant=do_quant,
            target=args.target,
        )
        return 0
    except (ModelLoadError, ConfigurationError, ValueError) as exc:
        print(f"Conversion failed: {exc}")
        return 1
    except KeyboardInterrupt:
        print("Interrupted by user")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
