#!/usr/bin/env python3
"""Unified configuration loader with clear priority chain.

This module implements a systematic configuration loading strategy:
  CLI args > Environment variables > YAML config > Python defaults

Priority Chain (highest to lowest):
1. CLI arguments (argparse) - Runtime override
2. Environment variables (RK_*) - System-level config
3. YAML config files (config/*.yaml) - Project-level config
4. Python constants (apps/config.py) - Code defaults

Usage:
    from apps.config_loader import ConfigLoader

    # Method 1: Load configuration with defaults
    loader = ConfigLoader()
    config = loader.load(
        cli_args={'imgsz': 640},
        config_file='config/app.yaml'
    )

    # Method 2: Get specific value with priority chain
    imgsz = loader.get('imgsz', default=416)

Example:
    # If user sets:
    # - CLI: --imgsz 640
    # - ENV: RK_IMGSZ=416
    # - YAML: imgsz: 320
    # - Python: DEFAULT_SIZE = 416
    # Result: imgsz=640 (CLI wins)
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union

from apps.config import (
    ModelConfig,
    RKNNConfig,
    PreprocessConfig,
    VideoConfig,
    NetworkConfig,
    PathConfig,
    CalibrationConfig,
    PerformanceConfig,
)
from apps.logger import setup_logger
from apps.exceptions import ConfigurationError, ValidationError

logger = setup_logger(__name__, level='INFO')


class ConfigLoader:
    """Unified configuration loader with priority chain.

    Priority: CLI > ENV > YAML > Python defaults
    """

    # Environment variable prefix
    ENV_PREFIX = 'RK_'

    # Default config file
    DEFAULT_CONFIG_FILE = 'config/app.yaml'

    def __init__(self, config_file: Optional[str] = None):
        """Initialize config loader.

        Args:
            config_file: Path to YAML config file (optional)
        """
        self.config_file = config_file or self.DEFAULT_CONFIG_FILE
        self._yaml_config = {}
        self._load_yaml_config()

    def _load_yaml_config(self):
        """Load YAML configuration file if it exists."""
        config_path = Path(self.config_file)

        if not config_path.exists():
            logger.debug(f"YAML config not found: {self.config_file} (using defaults)")
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._yaml_config = yaml.safe_load(f) or {}
            logger.debug(f"Loaded YAML config from: {self.config_file}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {self.config_file}: {e}") from e
        except (IOError, OSError) as e:
            raise ConfigurationError(f"Failed to read {self.config_file}: {e}") from e

    def get(
        self,
        key: str,
        cli_value: Any = None,
        default: Any = None,
        value_type: type = None,
        validate: callable = None,
    ) -> Any:
        """Get configuration value using priority chain.

        Args:
            key: Configuration key name
            cli_value: CLI argument value (highest priority)
            default: Default value (lowest priority)
            value_type: Expected value type for validation
            validate: Validation function (raises ValueError if invalid)

        Returns:
            Configuration value from highest priority source

        Raises:
            ValidationError: If value fails validation
        """
        source = None
        value = None

        # Priority 1: CLI argument
        if cli_value is not None:
            value = cli_value
            source = 'CLI'

        # Priority 2: Environment variable
        elif (env_key := f"{self.ENV_PREFIX}{key.upper()}") in os.environ:
            value = os.environ[env_key]
            source = 'ENV'

            # Auto-convert numeric strings
            if value_type in (int, float):
                try:
                    value = value_type(value)
                except ValueError as e:
                    raise ValidationError(
                        f"Invalid {value_type.__name__} value for {env_key}={value}: {e}"
                    ) from e

        # Priority 3: YAML config
        elif key in self._yaml_config:
            value = self._yaml_config[key]
            source = 'YAML'

        # Priority 4: Python default
        else:
            value = default
            source = 'DEFAULT'

        # Type validation
        if value_type is not None and value is not None:
            if not isinstance(value, value_type):
                try:
                    value = value_type(value)
                except (ValueError, TypeError) as e:
                    raise ValidationError(
                        f"Cannot convert {key}={value} to {value_type.__name__}: {e}"
                    ) from e

        # Custom validation
        if validate is not None and value is not None:
            try:
                validate(value)
            except ValueError as e:
                raise ValidationError(f"Validation failed for {key}={value}: {e}") from e

        logger.debug(f"Config[{key}] = {value} (source: {source})")
        return value

    def get_model_config(
        self,
        imgsz: Optional[int] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Get model configuration with priority chain.

        Args:
            imgsz: Image size from CLI (optional)
            conf_threshold: Confidence threshold from CLI (optional)
            iou_threshold: IOU threshold from CLI (optional)

        Returns:
            Model configuration dictionary
        """
        # Image size
        size = self.get(
            'imgsz',
            cli_value=imgsz,
            default=ModelConfig.DEFAULT_SIZE,
            value_type=int,
            validate=lambda x: x in [416, 640] or (_ for _ in ()).throw(
                ValueError(f"Image size must be 416 or 640, got {x}")
            )
        )

        # Confidence threshold
        conf = self.get(
            'conf_threshold',
            cli_value=conf_threshold,
            default=ModelConfig.CONF_THRESHOLD_DEFAULT,
            value_type=float,
            validate=lambda x: 0.0 < x < 1.0 or (_ for _ in ()).throw(
                ValueError(f"Confidence threshold must be in (0, 1), got {x}")
            )
        )

        # IOU threshold
        iou = self.get(
            'iou_threshold',
            cli_value=iou_threshold,
            default=ModelConfig.IOU_THRESHOLD_DEFAULT,
            value_type=float,
            validate=lambda x: 0.0 < x < 1.0 or (_ for _ in ()).throw(
                ValueError(f"IOU threshold must be in (0, 1), got {x}")
            )
        )

        # Max detections based on size
        max_det = (
            ModelConfig.MAX_DETECTIONS_416 if size == 416
            else ModelConfig.MAX_DETECTIONS_640
        )

        return {
            'imgsz': size,
            'conf_threshold': conf,
            'iou_threshold': iou,
            'max_detections': max_det,
        }

    def get_rknn_config(
        self,
        target_platform: Optional[str] = None,
        optimization_level: Optional[int] = None,
        core_mask: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get RKNN configuration with priority chain.

        Args:
            target_platform: Target platform from CLI (optional)
            optimization_level: Optimization level from CLI (optional)
            core_mask: NPU core mask from CLI (optional)

        Returns:
            RKNN configuration dictionary
        """
        platform = self.get(
            'target_platform',
            cli_value=target_platform,
            default=RKNNConfig.TARGET_PLATFORM,
            value_type=str,
        )

        opt_level = self.get(
            'optimization_level',
            cli_value=optimization_level,
            default=RKNNConfig.OPTIMIZATION_LEVEL,
            value_type=int,
            validate=lambda x: 0 <= x <= 3 or (_ for _ in ()).throw(
                ValueError(f"Optimization level must be 0-3, got {x}")
            )
        )

        mask = self.get(
            'core_mask',
            cli_value=core_mask,
            default=RKNNConfig.CORE_MASK_ALL,
            value_type=int,
        )

        return {
            'target_platform': platform,
            'optimization_level': opt_level,
            'core_mask': mask,
        }

    def get_log_level(self, cli_value: Optional[str] = None) -> str:
        """Get logging level with priority chain.

        Args:
            cli_value: Log level from CLI (optional)

        Returns:
            Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        level = self.get(
            'log_level',
            cli_value=cli_value,
            default='INFO',
            value_type=str,
            validate=lambda x: x.upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] or (
                _ for _ in ()
            ).throw(ValueError(f"Invalid log level: {x}"))
        )
        return level.upper()

    def dump_config(self, config: Dict[str, Any]) -> str:
        """Dump configuration as formatted string for logging.

        Args:
            config: Configuration dictionary

        Returns:
            Formatted configuration string
        """
        lines = ["Configuration:"]
        for key, value in sorted(config.items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


# Singleton instance for convenience
_default_loader = None


def get_loader(config_file: Optional[str] = None) -> ConfigLoader:
    """Get default config loader instance (singleton).

    Args:
        config_file: Path to YAML config file (optional)

    Returns:
        ConfigLoader instance
    """
    global _default_loader
    if _default_loader is None or config_file is not None:
        _default_loader = ConfigLoader(config_file=config_file)
    return _default_loader
