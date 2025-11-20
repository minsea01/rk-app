"""Model metadata extraction and management.

This module provides utilities to extract and manage model metadata including:
- DFL regression maximum value (reg_max)
- Feature map strides
- Head type (DFL vs raw)
- Number of classes

Supports reading from:
1. ONNX model metadata
2. RKNN model sidecar JSON files
3. YAML configuration files
4. Heuristic inference as fallback
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from apps.exceptions import ModelLoadError, ValidationError
from apps.logger import get_logger

logger = get_logger(__name__)


class ModelMetadata:
    """Container for YOLO model metadata."""

    def __init__(
        self,
        reg_max: int = 16,
        strides: List[int] = None,
        head_type: str = 'dfl',
        num_classes: int = 80,
        input_size: Tuple[int, int] = (640, 640),
        source: str = 'default'
    ):
        """Initialize model metadata.

        Args:
            reg_max: DFL regression maximum value (default: 16)
            strides: Feature map strides (default: [8, 16, 32])
            head_type: Head decode type ('dfl' or 'raw')
            num_classes: Number of detection classes
            input_size: Model input size (H, W)
            source: Source of metadata ('onnx', 'json', 'yaml', 'heuristic', 'default')
        """
        self.reg_max = reg_max
        self.strides = strides or [8, 16, 32]
        self.head_type = head_type
        self.num_classes = num_classes
        self.input_size = input_size
        self.source = source

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'reg_max': self.reg_max,
            'strides': self.strides,
            'head_type': self.head_type,
            'num_classes': self.num_classes,
            'input_size': list(self.input_size),
            'source': self.source
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelMetadata':
        """Create from dictionary."""
        return cls(
            reg_max=data.get('reg_max', 16),
            strides=data.get('strides', [8, 16, 32]),
            head_type=data.get('head_type', 'dfl'),
            num_classes=data.get('num_classes', 80),
            input_size=tuple(data.get('input_size', [640, 640])),
            source=data.get('source', 'dict')
        )

    def save_json(self, path: Union[str, Path]) -> None:
        """Save metadata to JSON file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved model metadata to {path}")

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> 'ModelMetadata':
        """Load metadata from JSON file."""
        path = Path(path)
        if not path.exists():
            raise ModelLoadError(f"Metadata file not found: {path}")
        with open(path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded model metadata from {path} (source: {data.get('source', 'json')})")
        return cls.from_dict({**data, 'source': 'json'})

    @classmethod
    def load_yaml(cls, path: Union[str, Path]) -> 'ModelMetadata':
        """Load metadata from YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")

        path = Path(path)
        if not path.exists():
            raise ModelLoadError(f"YAML config file not found: {path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Extract model metadata from config
        model_cfg = data.get('model', {})
        logger.info(f"Loaded model metadata from YAML {path}")
        return cls(
            reg_max=model_cfg.get('reg_max', 16),
            strides=model_cfg.get('strides', [8, 16, 32]),
            head_type=model_cfg.get('head_type', 'dfl'),
            num_classes=model_cfg.get('num_classes', 80),
            input_size=tuple(model_cfg.get('input_size', [640, 640])),
            source='yaml'
        )


def extract_onnx_metadata(model_path: Union[str, Path]) -> ModelMetadata:
    """Extract metadata from ONNX model.

    Args:
        model_path: Path to ONNX model file

    Returns:
        ModelMetadata object

    Raises:
        ModelLoadError: If model cannot be loaded
        ImportError: If onnx package not installed
    """
    if not HAS_ONNX:
        raise ImportError("onnx package not installed. Install with: pip install onnx")

    model_path = Path(model_path)
    if not model_path.exists():
        raise ModelLoadError(f"ONNX model not found: {model_path}")

    try:
        model = onnx.load(str(model_path))
    except Exception as e:
        raise ModelLoadError(f"Failed to load ONNX model: {e}") from e

    # Try to extract from metadata_props
    metadata = {}
    for prop in model.metadata_props:
        metadata[prop.key] = prop.value

    # Try to extract from model structure
    # Analyze output shape to infer parameters
    output_shape = None
    if model.graph.output:
        output = model.graph.output[0]
        if hasattr(output.type, 'tensor_type') and output.type.tensor_type.shape:
            dims = output.type.tensor_type.shape.dim
            output_shape = [d.dim_value if d.dim_value > 0 else None for d in dims]

    # Infer parameters from output shape
    reg_max = int(metadata.get('reg_max', 16))
    strides = json.loads(metadata.get('strides', '[8, 16, 32]'))
    head_type = metadata.get('head_type', 'dfl')
    num_classes = int(metadata.get('num_classes', 80))

    # Heuristic: if output channel count suggests DFL head
    if output_shape and len(output_shape) >= 2:
        c = output_shape[-1] if output_shape[-1] else output_shape[1]
        if c and c >= 64:
            # Likely DFL: channel = 4*reg_max + num_classes
            inferred_nc = c - 4 * reg_max
            if inferred_nc > 0 and 'num_classes' not in metadata:
                num_classes = inferred_nc
                logger.info(f"Inferred num_classes={num_classes} from output shape")

    logger.info(f"Extracted ONNX metadata: reg_max={reg_max}, strides={strides}, head={head_type}")

    return ModelMetadata(
        reg_max=reg_max,
        strides=strides,
        head_type=head_type,
        num_classes=num_classes,
        source='onnx'
    )


def infer_from_output_shape(
    output_shape: Tuple[int, ...],
    imgsz: int = 640
) -> ModelMetadata:
    """Infer model metadata from output tensor shape (heuristic fallback).

    Args:
        output_shape: Output tensor shape (B, C, N) or (B, N, C)
        imgsz: Input image size

    Returns:
        ModelMetadata with inferred parameters
    """
    if len(output_shape) < 2:
        logger.warning("Invalid output shape for inference, using defaults")
        return ModelMetadata(source='heuristic_failed')

    # Normalize to (C, N) or (N, C)
    if len(output_shape) == 3:
        shape = output_shape[1:]
    else:
        shape = output_shape

    # Determine (C, N) ordering
    c, n = shape if shape[0] < shape[1] else (shape[1], shape[0])

    # Infer head type from channel count
    if c >= 64:
        # DFL head: C = 4*reg_max + num_classes
        # Try common reg_max values
        for reg_max_candidate in [16, 8, 15, 12]:
            nc = c - 4 * reg_max_candidate
            if nc > 0 and nc <= 1000:  # Reasonable class count
                logger.info(
                    f"Heuristic: inferred reg_max={reg_max_candidate}, "
                    f"num_classes={nc} from output shape {output_shape}"
                )
                return ModelMetadata(
                    reg_max=reg_max_candidate,
                    strides=[8, 16, 32],
                    head_type='dfl',
                    num_classes=nc,
                    source='heuristic'
                )

    # Raw head: C = 5 + num_classes (cx, cy, w, h, obj, cls...)
    if c >= 5:
        nc = c - 5
        logger.info(
            f"Heuristic: inferred raw head with num_classes={nc} "
            f"from output shape {output_shape}"
        )
        return ModelMetadata(
            reg_max=16,  # Not used for raw head
            strides=[8, 16, 32],
            head_type='raw',
            num_classes=nc,
            source='heuristic'
        )

    # Fallback to defaults
    logger.warning(f"Could not infer metadata from shape {output_shape}, using defaults")
    return ModelMetadata(source='heuristic_failed')


def load_model_metadata(
    model_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    force_heuristic: bool = False
) -> ModelMetadata:
    """Load model metadata from available sources.

    Priority order:
    1. Explicit config file (JSON or YAML) if provided
    2. Sidecar JSON file (model_path + '.meta.json')
    3. ONNX model metadata (if model is .onnx)
    4. Heuristic inference (fallback)

    Args:
        model_path: Path to model file (.onnx or .rknn)
        config_path: Optional path to metadata config file (.json or .yaml)
        force_heuristic: If True, skip loading and use heuristic inference

    Returns:
        ModelMetadata object
    """
    model_path = Path(model_path)

    if force_heuristic:
        logger.info("force_heuristic=True, skipping metadata loading")
        return ModelMetadata(source='forced_default')

    # 1. Try explicit config file
    if config_path:
        config_path = Path(config_path)
        try:
            if config_path.suffix == '.json':
                return ModelMetadata.load_json(config_path)
            elif config_path.suffix in ['.yaml', '.yml']:
                return ModelMetadata.load_yaml(config_path)
            else:
                logger.warning(f"Unknown config file extension: {config_path.suffix}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")

    # 2. Try sidecar JSON
    sidecar_json = model_path.with_suffix(model_path.suffix + '.meta.json')
    if sidecar_json.exists():
        try:
            return ModelMetadata.load_json(sidecar_json)
        except Exception as e:
            logger.warning(f"Failed to load sidecar metadata {sidecar_json}: {e}")

    # 3. Try ONNX metadata extraction
    if model_path.suffix == '.onnx':
        try:
            return extract_onnx_metadata(model_path)
        except Exception as e:
            logger.warning(f"Failed to extract ONNX metadata: {e}")

    # 4. Fallback to defaults with warning
    logger.warning(
        f"No metadata found for {model_path.name}. "
        f"Using defaults (reg_max=16, strides=[8,16,32], head='dfl'). "
        f"Create {sidecar_json.name} to specify custom metadata."
    )
    return ModelMetadata(source='default_fallback')


if __name__ == '__main__':
    # Example usage
    import sys
    if len(sys.argv) > 1:
        model_path = Path(sys.argv[1])
        meta = load_model_metadata(model_path)
        print(json.dumps(meta.to_dict(), indent=2))
    else:
        print("Usage: python -m apps.model_meta <model_path>")
