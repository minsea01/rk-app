#!/usr/bin/env python3
"""Unit tests for tools.convert_onnx_to_rknn module.

Tests ONNX to RKNN conversion with validation of parameters and error handling.
"""
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tools.convert_onnx_to_rknn import (
    _detect_rknn_default_qdtype,
    _parse_and_validate_mean_std,
    build_rknn,
)
from apps.exceptions import ConfigurationError, ModelLoadError


class TestDetectRKNNDefaultQdtype:
    """Test suite for RKNN quantization dtype detection."""

    def test_returns_w8a8_for_rknn_toolkit2_v2_or_higher(self):
        """Test that w8a8 is returned for rknn-toolkit2 version 2.x."""
        with patch('tools.convert_onnx_to_rknn.version', return_value='2.3.2'):
            result = _detect_rknn_default_qdtype()
            assert result == 'w8a8'

    def test_returns_w8a8_for_rknn_toolkit2_v3(self):
        """Test that w8a8 is returned for rknn-toolkit2 version 3.x."""
        with patch('tools.convert_onnx_to_rknn.version', return_value='3.0.0'):
            result = _detect_rknn_default_qdtype()
            assert result == 'w8a8'

    def test_returns_asymmetric_for_rknn_toolkit2_v1(self):
        """Test that asymmetric_quantized-u8 is returned for version 1.x."""
        with patch('tools.convert_onnx_to_rknn.version', return_value='1.7.3'):
            result = _detect_rknn_default_qdtype()
            assert result == 'asymmetric_quantized-u8'

    def test_handles_missing_package_gracefully(self):
        """Test that missing package returns fallback dtype."""
        from importlib.metadata import PackageNotFoundError
        with patch('tools.convert_onnx_to_rknn.version', side_effect=PackageNotFoundError):
            result = _detect_rknn_default_qdtype()
            # Should return asymmetric for version 0 (fallback)
            assert result == 'asymmetric_quantized-u8'

    def test_handles_invalid_version_string(self):
        """Test that invalid version strings are handled gracefully."""
        with patch('tools.convert_onnx_to_rknn.version', return_value='invalid.version'):
            result = _detect_rknn_default_qdtype()
            # Should return asymmetric for non-numeric major version
            assert result == 'asymmetric_quantized-u8'


class TestParseMeanStd:
    """Test suite for mean/std parameter parsing and validation."""

    def test_parses_valid_mean_std_strings(self):
        """Test that valid comma-separated values are parsed correctly."""
        mean_vals, std_vals = _parse_and_validate_mean_std('0,0,0', '255,255,255')

        assert mean_vals == [[0.0, 0.0, 0.0]]
        assert std_vals == [[255.0, 255.0, 255.0]]

    def test_parses_float_values(self):
        """Test that float values are parsed correctly."""
        mean_vals, std_vals = _parse_and_validate_mean_std('123.45,67.89,0.1', '255.5,128.25,1.0')

        assert mean_vals == [[123.45, 67.89, 0.1]]
        assert std_vals == [[255.5, 128.25, 1.0]]

    def test_handles_whitespace_in_values(self):
        """Test that whitespace is stripped from values."""
        mean_vals, std_vals = _parse_and_validate_mean_std(' 0 , 0 , 0 ', ' 255 , 255 , 255 ')

        assert mean_vals == [[0.0, 0.0, 0.0]]
        assert std_vals == [[255.0, 255.0, 255.0]]

    def test_validates_three_mean_values_required(self):
        """Test that exactly 3 mean values are required."""
        with pytest.raises(ValueError, match="Mean must have exactly 3 values"):
            _parse_and_validate_mean_std('0,0', '255,255,255')

        with pytest.raises(ValueError, match="Mean must have exactly 3 values"):
            _parse_and_validate_mean_std('0,0,0,0', '255,255,255')

    def test_validates_three_std_values_required(self):
        """Test that exactly 3 std values are required."""
        with pytest.raises(ValueError, match="Std must have exactly 3 values"):
            _parse_and_validate_mean_std('0,0,0', '255,255')

        with pytest.raises(ValueError, match="Std must have exactly 3 values"):
            _parse_and_validate_mean_std('0,0,0', '255,255,255,255')

    def test_rejects_zero_std_values(self):
        """Test that std values of zero are rejected (division by zero)."""
        with pytest.raises(ValueError, match="Std values cannot be zero"):
            _parse_and_validate_mean_std('0,0,0', '255,0,255')

        with pytest.raises(ValueError, match="Std values cannot be zero"):
            _parse_and_validate_mean_std('0,0,0', '0,0,0')

    def test_rejects_negative_std_values(self):
        """Test that negative std values are rejected."""
        with pytest.raises(ValueError, match="Std values must be positive"):
            _parse_and_validate_mean_std('0,0,0', '255,-128,255')

        with pytest.raises(ValueError, match="Std values must be positive"):
            _parse_and_validate_mean_std('0,0,0', '-1,-1,-1')

    def test_raises_on_invalid_float_format(self):
        """Test that invalid float strings raise ValueError."""
        with pytest.raises(ValueError, match="Invalid mean/std format"):
            _parse_and_validate_mean_std('a,b,c', '255,255,255')

        with pytest.raises(ValueError, match="Invalid mean/std format"):
            _parse_and_validate_mean_std('0,0,0', 'x,y,z')

    def test_handles_negative_mean_values(self):
        """Test that negative mean values are allowed (valid for normalization)."""
        mean_vals, std_vals = _parse_and_validate_mean_std('-0.485,-0.456,-0.406', '0.229,0.224,0.225')

        assert mean_vals == [[-0.485, -0.456, -0.406]]
        assert std_vals == [[0.229, 0.224, 0.225]]


class TestBuildRKNN:
    """Test suite for build_rknn function."""

    def setup_method(self):
        """Create temporary directories for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def test_raises_configuration_error_when_rknn_toolkit_not_installed(self):
        """Test that ConfigurationError is raised when rknn-toolkit2 is missing."""
        onnx_path = self.temp_path / 'model.onnx'
        onnx_path.touch()
        out_path = self.temp_path / 'model.rknn'

        with patch('tools.convert_onnx_to_rknn.RKNN', side_effect=ImportError("No module named 'rknn'")):
            with pytest.raises(ConfigurationError, match="rknn-toolkit2 not installed"):
                build_rknn(onnx_path, out_path)

    def test_raises_configuration_error_on_rknn_toolkit_incompatibility(self):
        """Test that ConfigurationError is raised for incompatible rknn-toolkit2."""
        onnx_path = self.temp_path / 'model.onnx'
        onnx_path.touch()
        out_path = self.temp_path / 'model.rknn'

        with patch('tools.convert_onnx_to_rknn.RKNN', side_effect=AttributeError("Incompatible version")):
            with pytest.raises(ConfigurationError, match="rknn-toolkit2 version incompatible"):
                build_rknn(onnx_path, out_path)

    def test_validates_mean_std_parameters(self):
        """Test that mean/std parameters are validated."""
        onnx_path = self.temp_path / 'model.onnx'
        onnx_path.touch()
        out_path = self.temp_path / 'model.rknn'

        # Mock RKNN class
        mock_rknn_class = MagicMock()
        mock_rknn_instance = MagicMock()
        mock_rknn_class.return_value = mock_rknn_instance

        with patch('tools.convert_onnx_to_rknn.RKNN', mock_rknn_class):
            # Should raise ValueError for invalid mean
            with pytest.raises(ValueError, match="Mean must have exactly 3 values"):
                build_rknn(onnx_path, out_path, mean='0,0', std='255,255,255')

            # Should raise ValueError for zero std
            with pytest.raises(ValueError, match="Std values cannot be zero"):
                build_rknn(onnx_path, out_path, mean='0,0,0', std='255,0,255')

    def test_calls_rknn_api_correctly(self):
        """Test that RKNN API methods are called with correct parameters."""
        onnx_path = self.temp_path / 'model.onnx'
        onnx_path.write_text('fake onnx content')
        out_path = self.temp_path / 'model.rknn'

        # Mock RKNN class and instance
        mock_rknn_class = MagicMock()
        mock_rknn_instance = MagicMock()
        mock_rknn_class.return_value = mock_rknn_instance

        # Mock successful operations
        mock_rknn_instance.config.return_value = 0
        mock_rknn_instance.load_onnx.return_value = 0
        mock_rknn_instance.build.return_value = 0
        mock_rknn_instance.export_rknn.return_value = 0

        with patch('tools.convert_onnx_to_rknn.RKNN', mock_rknn_class):
            build_rknn(
                onnx_path,
                out_path,
                do_quant=True,
                target='rk3588',
                mean='0,0,0',
                std='255,255,255'
            )

            # Verify API calls
            mock_rknn_instance.config.assert_called_once()
            mock_rknn_instance.load_onnx.assert_called_once_with(str(onnx_path))
            mock_rknn_instance.build.assert_called_once()
            mock_rknn_instance.export_rknn.assert_called_once_with(str(out_path))
            mock_rknn_instance.release.assert_called_once()

    def test_handles_calibration_dataset_path(self):
        """Test that calibration dataset path is validated and used."""
        onnx_path = self.temp_path / 'model.onnx'
        onnx_path.write_text('fake onnx')
        out_path = self.temp_path / 'model.rknn'
        calib_path = self.temp_path / 'calib.txt'
        calib_path.write_text('/path/to/image1.jpg\n/path/to/image2.jpg\n')

        mock_rknn_class = MagicMock()
        mock_rknn_instance = MagicMock()
        mock_rknn_class.return_value = mock_rknn_instance

        # Mock successful operations
        mock_rknn_instance.config.return_value = 0
        mock_rknn_instance.load_onnx.return_value = 0
        mock_rknn_instance.build.return_value = 0
        mock_rknn_instance.export_rknn.return_value = 0

        with patch('tools.convert_onnx_to_rknn.RKNN', mock_rknn_class):
            build_rknn(
                onnx_path,
                out_path,
                calib=calib_path,
                do_quant=True
            )

            # Verify that build was called with dataset parameter
            mock_rknn_instance.build.assert_called_once()
            call_kwargs = mock_rknn_instance.build.call_args[1]
            assert 'dataset' in call_kwargs
            assert call_kwargs['dataset'] == str(calib_path)

    def test_creates_output_directory_if_needed(self):
        """Test that output directory is created if it doesn't exist."""
        onnx_path = self.temp_path / 'model.onnx'
        onnx_path.write_text('fake onnx')

        # Output path in non-existent subdirectory
        out_path = self.temp_path / 'output' / 'subdir' / 'model.rknn'

        mock_rknn_class = MagicMock()
        mock_rknn_instance = MagicMock()
        mock_rknn_class.return_value = mock_rknn_instance

        mock_rknn_instance.config.return_value = 0
        mock_rknn_instance.load_onnx.return_value = 0
        mock_rknn_instance.build.return_value = 0
        mock_rknn_instance.export_rknn.return_value = 0

        with patch('tools.convert_onnx_to_rknn.RKNN', mock_rknn_class):
            build_rknn(onnx_path, out_path)

            # Parent directory should be created
            assert out_path.parent.exists()

    def test_uses_custom_quantization_dtype(self):
        """Test that custom quantization dtype is used when specified."""
        onnx_path = self.temp_path / 'model.onnx'
        onnx_path.write_text('fake onnx')
        out_path = self.temp_path / 'model.rknn'

        mock_rknn_class = MagicMock()
        mock_rknn_instance = MagicMock()
        mock_rknn_class.return_value = mock_rknn_instance

        mock_rknn_instance.config.return_value = 0
        mock_rknn_instance.load_onnx.return_value = 0
        mock_rknn_instance.build.return_value = 0
        mock_rknn_instance.export_rknn.return_value = 0

        with patch('tools.convert_onnx_to_rknn.RKNN', mock_rknn_class):
            build_rknn(onnx_path, out_path, quantized_dtype='asymmetric_quantized-u8')

            # Verify that build was called with custom dtype
            call_kwargs = mock_rknn_instance.build.call_args[1]
            assert 'quantized_dtype' in call_kwargs
            assert call_kwargs['quantized_dtype'] == 'asymmetric_quantized-u8'

    def test_skips_quantization_when_do_quant_false(self):
        """Test that quantization is skipped when do_quant=False."""
        onnx_path = self.temp_path / 'model.onnx'
        onnx_path.write_text('fake onnx')
        out_path = self.temp_path / 'model.rknn'

        mock_rknn_class = MagicMock()
        mock_rknn_instance = MagicMock()
        mock_rknn_class.return_value = mock_rknn_instance

        mock_rknn_instance.config.return_value = 0
        mock_rknn_instance.load_onnx.return_value = 0
        mock_rknn_instance.build.return_value = 0
        mock_rknn_instance.export_rknn.return_value = 0

        with patch('tools.convert_onnx_to_rknn.RKNN', mock_rknn_class):
            build_rknn(onnx_path, out_path, do_quant=False)

            # Verify that build was called without quantization parameters
            call_kwargs = mock_rknn_instance.build.call_args[1]
            assert call_kwargs.get('do_quantization') is False
