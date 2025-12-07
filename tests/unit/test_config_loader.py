#!/usr/bin/env python3
"""Unit tests for apps/config_loader.py

Tests the configuration priority chain: CLI > ENV > YAML > Python defaults
"""
import os
import pytest
import tempfile
from pathlib import Path

from apps.config_loader import ConfigLoader, get_loader, _default_loader
import apps.config_loader as config_loader_module
from apps.exceptions import ConfigurationError, ValidationError
from apps.config import ModelConfig


class TestConfigLoader:
    """Test ConfigLoader priority chain and validation."""

    def test_priority_chain_cli_wins(self):
        """Test that CLI arguments have highest priority."""
        loader = ConfigLoader()

        # Set all levels
        os.environ['RK_IMGSZ'] = '416'  # ENV
        # YAML would be loaded from file (skipped in this test)

        result = loader.get(
            'imgsz',
            cli_value=640,  # CLI (should win)
            default=320,    # Python default
            value_type=int
        )

        assert result == 640, "CLI should have highest priority"

        # Cleanup
        del os.environ['RK_IMGSZ']

    def test_priority_chain_env_wins_over_default(self):
        """Test that ENV wins over Python defaults when no CLI."""
        loader = ConfigLoader()

        os.environ['RK_IMGSZ'] = '640'

        result = loader.get(
            'imgsz',
            cli_value=None,  # No CLI
            default=416,     # Default
            value_type=int
        )

        assert result == 640, "ENV should win over defaults"

        # Cleanup
        del os.environ['RK_IMGSZ']

    def test_priority_chain_default_used(self):
        """Test that default is used when no higher priority source."""
        loader = ConfigLoader()

        result = loader.get(
            'imgsz',
            cli_value=None,  # No CLI
            default=416,     # Should be used
            value_type=int
        )

        assert result == 416, "Default should be used when no other source"

    def test_yaml_config_loading(self):
        """Test loading configuration from YAML file."""
        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('imgsz: 640\nconf_threshold: 0.25\n')
            yaml_path = f.name

        try:
            loader = ConfigLoader(config_file=yaml_path)

            # YAML value should be used
            imgsz = loader.get('imgsz', default=416, value_type=int)
            assert imgsz == 640, "YAML config should be loaded"

            conf = loader.get('conf_threshold', default=0.5, value_type=float)
            assert conf == 0.25, "YAML config should be loaded"
        finally:
            Path(yaml_path).unlink()

    def test_invalid_yaml_raises_error(self):
        """Test that invalid YAML raises ConfigurationError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: syntax:\n')
            yaml_path = f.name

        try:
            with pytest.raises(ConfigurationError, match="Invalid YAML"):
                ConfigLoader(config_file=yaml_path)
        finally:
            Path(yaml_path).unlink()

    def test_type_conversion(self):
        """Test automatic type conversion."""
        loader = ConfigLoader()

        # String to int
        os.environ['RK_IMGSZ'] = '640'
        result = loader.get('imgsz', value_type=int)
        assert result == 640 and isinstance(result, int)
        del os.environ['RK_IMGSZ']

        # String to float
        os.environ['RK_CONF_THRESHOLD'] = '0.5'
        result = loader.get('conf_threshold', value_type=float)
        assert result == 0.5 and isinstance(result, float)
        del os.environ['RK_CONF_THRESHOLD']

    def test_type_validation_failure(self):
        """Test that invalid types raise ValidationError."""
        loader = ConfigLoader()

        os.environ['RK_IMGSZ'] = 'not_a_number'

        with pytest.raises(ValidationError, match="Invalid int value"):
            loader.get('imgsz', value_type=int)

        del os.environ['RK_IMGSZ']

    def test_custom_validation(self):
        """Test custom validation function."""
        loader = ConfigLoader()

        def validate_imgsz(value):
            if value not in [416, 640]:
                raise ValueError(f"Must be 416 or 640, got {value}")

        # Valid value
        result = loader.get('imgsz', default=416, validate=validate_imgsz)
        assert result == 416

        # Invalid value
        with pytest.raises(ValidationError, match="Validation failed"):
            loader.get('imgsz', default=999, validate=validate_imgsz)

    def test_get_model_config(self):
        """Test get_model_config helper method."""
        loader = ConfigLoader()

        # Default configuration
        config = loader.get_model_config()
        assert config['imgsz'] == ModelConfig.DEFAULT_SIZE
        assert config['conf_threshold'] == ModelConfig.CONF_THRESHOLD_DEFAULT
        assert 'max_detections' in config

        # With CLI override
        config = loader.get_model_config(imgsz=640, conf_threshold=0.25)
        assert config['imgsz'] == 640
        assert config['conf_threshold'] == 0.25
        assert config['max_detections'] == ModelConfig.MAX_DETECTIONS_640

    def test_get_model_config_validation(self):
        """Test that get_model_config validates parameters."""
        loader = ConfigLoader()

        # Invalid image size
        with pytest.raises(ValidationError, match="Image size must be 416 or 640"):
            loader.get_model_config(imgsz=320)

        # Invalid confidence threshold
        with pytest.raises(ValidationError, match="Confidence threshold must be in"):
            loader.get_model_config(conf_threshold=1.5)

    def test_get_rknn_config(self):
        """Test get_rknn_config helper method."""
        loader = ConfigLoader()

        config = loader.get_rknn_config()
        assert 'target_platform' in config
        assert 'optimization_level' in config
        assert 'core_mask' in config

        # With CLI override
        config = loader.get_rknn_config(optimization_level=2)
        assert config['optimization_level'] == 2

    def test_get_log_level(self):
        """Test get_log_level with validation."""
        loader = ConfigLoader()

        # Valid levels
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            result = loader.get_log_level(cli_value=level)
            assert result == level

        # Case insensitive
        result = loader.get_log_level(cli_value='debug')
        assert result == 'DEBUG'

        # Invalid level
        with pytest.raises(ValidationError, match="Invalid log level"):
            loader.get_log_level(cli_value='INVALID')

    def test_nonexistent_yaml_graceful_fallback(self):
        """Test that nonexistent YAML file doesn't crash."""
        loader = ConfigLoader(config_file='nonexistent.yaml')
        result = loader.get('imgsz', default=416)
        assert result == 416, "Should gracefully fall back to default"

    def test_dump_config(self):
        """Test dump_config formatting."""
        loader = ConfigLoader()

        config = {'imgsz': 416, 'conf_threshold': 0.5}
        output = loader.dump_config(config)

        assert 'Configuration:' in output
        assert 'imgsz: 416' in output
        assert 'conf_threshold: 0.5' in output


class TestConfigLoaderIntegration:
    """Integration tests for ConfigLoader."""

    def test_full_priority_chain(self):
        """Test complete priority chain: CLI > ENV > YAML > Default."""
        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('imgsz: 320\n')
            yaml_path = f.name

        try:
            loader = ConfigLoader(config_file=yaml_path)

            # Test 1: All sources present - CLI should win
            os.environ['RK_IMGSZ'] = '416'
            result = loader.get('imgsz', cli_value=640, default=416, value_type=int)
            assert result == 640, "CLI should have highest priority"

            # Test 2: No CLI - ENV should win
            result = loader.get('imgsz', cli_value=None, default=416, value_type=int)
            assert result == 416, "ENV should win over YAML"

            # Test 3: No CLI, no ENV - YAML should win
            del os.environ['RK_IMGSZ']
            result = loader.get('imgsz', cli_value=None, default=416, value_type=int)
            assert result == 320, "YAML should win over default"

            # Test 4: Only default
            result = loader.get('nonexistent_key', cli_value=None, default=999, value_type=int)
            assert result == 999, "Default should be used when no other source"

        finally:
            Path(yaml_path).unlink()
            if 'RK_IMGSZ' in os.environ:
                del os.environ['RK_IMGSZ']


class TestGetLoader:
    """Test suite for get_loader() singleton function."""

    def setup_method(self):
        """Reset global singleton before each test."""
        config_loader_module._default_loader = None

    def teardown_method(self):
        """Clean up after each test."""
        config_loader_module._default_loader = None

    def test_get_loader_returns_config_loader_instance(self):
        """Test that get_loader returns a ConfigLoader instance."""
        loader = get_loader()
        assert isinstance(loader, ConfigLoader)

    def test_get_loader_returns_singleton(self):
        """Test that get_loader returns the same instance on subsequent calls."""
        loader1 = get_loader()
        loader2 = get_loader()
        assert loader1 is loader2, "get_loader should return singleton instance"

    def test_get_loader_recreates_with_new_config_file(self):
        """Test that providing config_file creates new instance."""
        loader1 = get_loader()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('imgsz: 640\n')
            yaml_path = f.name

        try:
            loader2 = get_loader(config_file=yaml_path)
            # Should be a different instance
            assert loader2 is not loader1, "New config_file should create new instance"
            # Should load the new config
            assert loader2.get('imgsz', default=416, value_type=int) == 640
        finally:
            Path(yaml_path).unlink()

    def test_get_loader_uses_default_config_file(self):
        """Test that get_loader uses default config file path."""
        loader = get_loader()
        assert loader.config_file == ConfigLoader.DEFAULT_CONFIG_FILE


class TestConfigLoaderErrorPaths:
    """Test suite for ConfigLoader error handling paths."""

    def test_float_env_conversion_error(self):
        """Test that invalid float ENV values raise ValidationError."""
        loader = ConfigLoader()

        os.environ['RK_CONF_THRESHOLD'] = 'not_a_float'

        try:
            with pytest.raises(ValidationError, match="Invalid float value"):
                loader.get('conf_threshold', value_type=float)
        finally:
            del os.environ['RK_CONF_THRESHOLD']

    def test_float_env_hex_not_supported(self):
        """Test that hex strings for float raise ValidationError."""
        loader = ConfigLoader()

        os.environ['RK_CONF_THRESHOLD'] = '0x10'  # Hex not valid for float

        try:
            with pytest.raises(ValidationError, match="Invalid float value"):
                loader.get('conf_threshold', value_type=float)
        finally:
            del os.environ['RK_CONF_THRESHOLD']

    def test_get_rknn_config_invalid_optimization_level(self):
        """Test that invalid optimization level raises ValidationError."""
        loader = ConfigLoader()

        # Optimization level must be 0-3
        with pytest.raises(ValidationError, match="Optimization level must be 0-3"):
            loader.get_rknn_config(optimization_level=5)

    def test_get_rknn_config_negative_optimization_level(self):
        """Test that negative optimization level raises ValidationError."""
        loader = ConfigLoader()

        with pytest.raises(ValidationError, match="Optimization level must be 0-3"):
            loader.get_rknn_config(optimization_level=-1)

    def test_get_model_config_zero_conf_threshold(self):
        """Test that zero confidence threshold raises ValidationError."""
        loader = ConfigLoader()

        with pytest.raises(ValidationError, match="Confidence threshold must be in"):
            loader.get_model_config(conf_threshold=0.0)

    def test_get_model_config_zero_iou_threshold(self):
        """Test that zero IOU threshold raises ValidationError."""
        loader = ConfigLoader()

        with pytest.raises(ValidationError, match="IOU threshold must be in"):
            loader.get_model_config(iou_threshold=0.0)

    def test_yaml_type_conversion_error(self):
        """Test type conversion error from YAML values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('imgsz: "not_an_int"\n')
            yaml_path = f.name

        try:
            loader = ConfigLoader(config_file=yaml_path)
            with pytest.raises(ValidationError, match="Cannot convert"):
                loader.get('imgsz', value_type=int)
        finally:
            Path(yaml_path).unlink()

    def test_int_env_hex_format_supported(self):
        """Test that hex format is supported for int ENV values."""
        loader = ConfigLoader()

        os.environ['RK_CORE_MASK'] = '0x7'  # Hex format

        try:
            result = loader.get('core_mask', value_type=int)
            assert result == 7, "Hex format should be parsed correctly"
        finally:
            del os.environ['RK_CORE_MASK']
