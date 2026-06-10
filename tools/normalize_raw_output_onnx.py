#!/usr/bin/env python3
"""Normalize decoded RAW YOLO ONNX output boxes to [0,1].

Expected model output shape:
  - [1, C, N] or
  - [1, N, C]

where C is at least 5. The first 4 channels must be decoded
cx, cy, w, h box coordinates and the remaining channels are scores/classes.

The script divides the first 4 channels (cx, cy, w, h) by `imgsz` and writes
back a new ONNX whose single output stays shape-compatible but now contains
normalized coordinates plus the original score/class channels. This makes INT8
quantization practical because all output channels share a similar numeric range.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.exceptions import ConfigurationError, ModelLoadError
from apps.logger import setup_logger

logger = setup_logger(__name__, level="INFO")


def _read_output_shape(model: onnx.ModelProto) -> tuple[str, list[int], int, int]:
    if not model.graph.output:
        raise ConfigurationError("ONNX model has no graph outputs")

    output = model.graph.output[0]
    shape = []
    for dim in output.type.tensor_type.shape.dim:
        if dim.dim_value <= 0:
            raise ConfigurationError("ONNX output shape must be static")
        shape.append(int(dim.dim_value))

    if len(shape) != 3:
        raise ConfigurationError(f"Expected 3D output tensor, got shape={shape}")

    candidate_axes = [axis for axis in (1, 2) if 5 <= shape[axis] <= 512]
    if len(candidate_axes) != 1:
        raise ConfigurationError(
            "Expected exactly one output dimension to look like channels "
            f"(5..512), got shape={shape}"
        )

    channel_axis = candidate_axes[0]
    channel_count = shape[channel_axis]
    return output.name, shape, channel_axis, channel_count


def normalize_raw_output(onnx_path: Path, out_path: Path, imgsz: int) -> Path:
    if imgsz <= 0:
        raise ConfigurationError(f"imgsz must be positive, got {imgsz}")
    if not onnx_path.exists():
        raise ModelLoadError(f"ONNX file not found: {onnx_path}")

    model = onnx.load(str(onnx_path))
    output_name, output_shape, channel_axis, channel_count = _read_output_shape(model)

    starts_boxes_name = "rkapp_norm_starts_boxes"
    ends_boxes_name = "rkapp_norm_ends_boxes"
    starts_scores_name = "rkapp_norm_starts_scores"
    ends_scores_name = "rkapp_norm_ends_scores"
    axes_name = "rkapp_norm_axes"
    steps_name = "rkapp_norm_steps"
    scale_name = "rkapp_norm_scale"

    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([0], dtype=np.int64), name=starts_boxes_name),
            numpy_helper.from_array(np.array([4], dtype=np.int64), name=ends_boxes_name),
            numpy_helper.from_array(np.array([4], dtype=np.int64), name=starts_scores_name),
            numpy_helper.from_array(
                np.array([channel_count], dtype=np.int64), name=ends_scores_name
            ),
            numpy_helper.from_array(np.array([channel_axis], dtype=np.int64), name=axes_name),
            numpy_helper.from_array(np.array([1], dtype=np.int64), name=steps_name),
            numpy_helper.from_array(np.array(float(imgsz), dtype=np.float32), name=scale_name),
        ]
    )

    boxes_name = "rkapp_norm_boxes"
    scores_name = "rkapp_norm_scores"
    norm_boxes_name = "rkapp_norm_boxes_scaled"
    normalized_output_name = "rkapp_norm_output"

    model.graph.node.extend(
        [
            helper.make_node(
                "Slice",
                inputs=[output_name, starts_boxes_name, ends_boxes_name, axes_name, steps_name],
                outputs=[boxes_name],
                name="RkappSliceBoxes",
            ),
            helper.make_node(
                "Slice",
                inputs=[output_name, starts_scores_name, ends_scores_name, axes_name, steps_name],
                outputs=[scores_name],
                name="RkappSliceScores",
            ),
            helper.make_node(
                "Div",
                inputs=[boxes_name, scale_name],
                outputs=[norm_boxes_name],
                name="RkappNormalizeBoxes",
            ),
            helper.make_node(
                "Concat",
                inputs=[norm_boxes_name, scores_name],
                outputs=[normalized_output_name],
                axis=channel_axis,
                name="RkappConcatNormalizedRawOutput",
            ),
        ]
    )

    del model.graph.output[:]
    model.graph.output.extend(
        [
            helper.make_tensor_value_info(
                normalized_output_name,
                TensorProto.FLOAT,
                output_shape,
            )
        ]
    )

    try:
        model = shape_inference.infer_shapes(model)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Shape inference failed, saving model without inferred shapes: %s", exc)

    onnx.checker.check_model(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    logger.info("Wrote normalized ONNX: %s", out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize decoded RAW YOLO ONNX output boxes")
    parser.add_argument("--onnx", type=Path, required=True, help="input ONNX path")
    parser.add_argument("--out", type=Path, required=True, help="output ONNX path")
    parser.add_argument("--imgsz", type=int, required=True, help="model input size")
    args = parser.parse_args()

    try:
        normalize_raw_output(args.onnx, args.out, args.imgsz)
        return 0
    except (ConfigurationError, ModelLoadError) as exc:
        logger.error("Failed to normalize ONNX output: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
