#!/usr/bin/env python3
"""Unit tests for apps/config_loader.py

Tests the configuration priority chain: CLI > ENV > YAML > Python defaults
"""
import os
import pytest
import tempfile
from pathlib import Path

from apps.config_loader import ConfigLoader
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
