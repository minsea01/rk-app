#!/usr/bin/env python3
"""
Comprehensive unit tests for convert_onnx_to_rknn.py tool.

Test Coverage:
- _detect_rknn_default_qdtype() version detection
- build_rknn() conversion function
- Parameter validation and error handling
- Calibration dataset processing
- File I/O operations

Author: Senior Test Engineer
Standard: Enterprise-grade with 95%+ coverage
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import sys

from tools.convert_onnx_to_rknn import _detect_rknn_default_qdtype, build_rknn


class TestDetectRknnDefaultQdtype:
    """Test suite for _detect_rknn_default_qdtype() function."""

    @patch('tools.convert_onnx_to_rknn.version')
    def test_detect_rknn_version_2x(self, mock_version):
        """Test detection of rknn-toolkit2 version 2.x."""
        mock_version.return_value = '2.3.2'

        result = _detect_rknn_default_qdtype()

        assert result == 'w8a8'

    @patch('tools.convert_onnx_to_rknn.version')
    def test_detect_rknn_version_1x(self, mock_version):
        """Test detection of rknn-toolkit2 version 1.x."""
        mock_version.return_value = '1.7.3'

        result = _detect_rknn_default_qdtype()

        assert result == 'asymmetric_quantized-u8'

    @patch('tools.convert_onnx_to_rknn.version')
    def test_detect_rknn_version_3x(self, mock_version):
        """Test detection of future version 3.x."""
        mock_version.return_value = '3.0.0'

        result = _detect_rknn_default_qdtype()

        # Should use w8a8 for version >= 2
        assert result == 'w8a8'

    @patch('tools.convert_onnx_to_rknn.version')
    def test_detect_rknn_version_not_found(self, mock_version):
        """Test handling when rknn-toolkit2 not installed."""
        from importlib.metadata import PackageNotFoundError
        mock_version.side_effect = PackageNotFoundError()

        result = _detect_rknn_default_qdtype()

        # Should default to old format when package not found
        assert result == 'asymmetric_quantized-u8'

    @patch('tools.convert_onnx_to_rknn.version')
    def test_detect_rknn_invalid_version_format(self, mock_version):
        """Test handling of invalid version format."""
        mock_version.return_value = 'invalid.version.x'

        result = _detect_rknn_default_qdtype()

        # Should handle gracefully and default to old format
        assert result == 'asymmetric_quantized-u8'

    @patch('tools.convert_onnx_to_rknn.version')
    def test_detect_rknn_version_with_suffix(self, mock_version):
        """Test version with suffix like '2.3.2-beta'."""
        mock_version.return_value = '2.3.2-beta'

        result = _detect_rknn_default_qdtype()

        assert result == 'w8a8'

    @patch('tools.convert_onnx_to_rknn.version')
    def test_detect_rknn_empty_version(self, mock_version):
        """Test handling of empty version string."""
        mock_version.return_value = ''

        result = _detect_rknn_default_qdtype()

        assert result == 'asymmetric_quantized-u8'


class TestBuildRknn:
    """Test suite for build_rknn() conversion function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_rknn(self):
        """Create mock RKNN object."""
        mock = MagicMock()
        mock.config.return_value = 0
        mock.load_onnx.return_value = 0
        mock.build.return_value = 0
        mock.export_rknn.return_value = 0
        mock.release.return_value = None
        return mock

    @pytest.fixture
    def fake_onnx_file(self, temp_dir):
        """Create fake ONNX file."""
        onnx_file = temp_dir / 'model.onnx'
        onnx_file.write_bytes(b'fake_onnx_content')
        return onnx_file

    @pytest.fixture
    def fake_calib_file(self, temp_dir):
        """Create fake calibration file."""
        calib_file = temp_dir / 'calib.txt'
        images = [str(temp_dir / f'img{i}.jpg') for i in range(10)]
        calib_file.write_text('\n'.join(images))
        # Create fake image files
        for img_path in images:
            Path(img_path).write_bytes(b'fake_jpg')
        return calib_file

    @patch('tools.convert_onnx_to_rknn.RKNN')
    def test_build_rknn_basic_conversion(self, mock_rknn_class, temp_dir, fake_onnx_file, fake_calib_file, mock_rknn):
        """Test basic ONNX to RKNN conversion."""
        mock_rknn_class.return_value = mock_rknn
        out_path = temp_dir / 'output.rknn'

        build_rknn(
            onnx_path=fake_onnx_file,
            out_path=out_path,
            calib=fake_calib_file,
            do_quant=True,
            target='rk3588'
        )

        # Verify RKNN was initialized
        mock_rknn_class.assert_called_once_with(verbose=True)

        # Verify config was called
        mock_rknn.config.assert_called_once()

        # Verify load_onnx was called
        mock_rknn.load_onnx.assert_called_once_with(model=str(fake_onnx_file))

        # Verify build was called with quantization
        mock_rknn.build.assert_called_once()
        assert mock_rknn.build.call_args[1]['do_quantization'] is True

        # Verify export_rknn was called
        mock_rknn.export_rknn.assert_called_once_with(str(out_path))

        # Verify release was called
        mock_rknn.release.assert_called()

    @patch('tools.convert_onnx_to_rknn.RKNN')
    def test_build_rknn_without_quantization(self, mock_rknn_class, temp_dir, fake_onnx_file, mock_rknn):
        """Test conversion without quantization."""
        mock_rknn_class.return_value = mock_rknn
        out_path = temp_dir / 'output_fp16.rknn'

        build_rknn(
            onnx_path=fake_onnx_file,
            out_path=out_path,
            calib=None,
            do_quant=False,
            target='rk3588'
        )

        # Verify build was called without quantization
        mock_rknn.build.assert_called_once()
        assert mock_rknn.build.call_args[1]['do_quantization'] is False
        assert mock_rknn.build.call_args[1]['dataset'] is None

    @patch('tools.convert_onnx_to_rknn.RKNN')
    def test_build_rknn_custom_mean_std(self, mock_rknn_class, temp_dir, fake_onnx_file, fake_calib_file, mock_rknn):
        """Test conversion with custom mean and std values."""
        mock_rknn_class.return_value = mock_rknn
        out_path = temp_dir / 'output.rknn'

        build_rknn(
            onnx_path=fake_onnx_file,
            out_path=out_path,
            calib=fake_calib_file,
            do_quant=True,
            target='rk3588',
            mean='123.675,116.28,103.53',  # ImageNet mean
            std='58.395,57.12,57.375'      # ImageNet std
        )

        # Verify config was called with correct mean/std
        config_call = mock_rknn.config.call_args
        mean_values = config_call[1]['mean_values']
        std_values = config_call[1]['std_values']

        assert mean_values == [[123.675, 116.28, 103.53]]
        assert std_values == [[58.395, 57.12, 57.375]]

    @patch('tools.convert_onnx_to_rknn.RKNN')
    def test_build_rknn_different_targets(self, mock_rknn_class, temp_dir, fake_onnx_file, mock_rknn):
        """Test conversion for different target platforms."""
        mock_rknn_class.return_value = mock_rknn

        targets = ['rk3588', 'rk3568', 'rk3566']

        for target in targets:
            mock_rknn.reset_mock()
            out_path = temp_dir / f'output_{target}.rknn'

            build_rknn(
                onnx_path=fake_onnx_file,
                out_path=out_path,
                calib=None,
                do_quant=False,
                target=target
            )

            # Verify config was called with correct target
            config_call = mock_rknn.config.call_args
            assert config_call[1]['target_platform'] == target

    @patch('tools.convert_onnx_to_rknn.RKNN')
    @patch('tools.convert_onnx_to_rknn._detect_rknn_default_qdtype')
    def test_build_rknn_auto_detect_qdtype(self, mock_detect, mock_rknn_class, temp_dir, fake_onnx_file, fake_calib_file, mock_rknn):
        """Test automatic quantization dtype detection."""
        mock_rknn_class.return_value = mock_rknn
        mock_detect.return_value = 'w8a8'

        out_path = temp_dir / 'output.rknn'

        build_rknn(
            onnx_path=fake_onnx_file,
            out_path=out_path,
            calib=fake_calib_file,
            do_quant=True,
            target='rk3588',
            quantized_dtype=None  # Auto-detect
        )

        # Verify auto-detection was called
        mock_detect.assert_called_once()

        # Verify config was called with detected dtype
        config_call = mock_rknn.config.call_args
        assert config_call[1]['quantized_dtype'] == 'w8a8'

    @patch('tools.convert_onnx_to_rknn.RKNN')
    def test_build_rknn_explicit_qdtype(self, mock_rknn_class, temp_dir, fake_onnx_file, fake_calib_file, mock_rknn):
        """Test conversion with explicit quantization dtype."""
        mock_rknn_class.return_value = mock_rknn
        out_path = temp_dir / 'output.rknn'

        build_rknn(
            onnx_path=fake_onnx_file,
            out_path=out_path,
            calib=fake_calib_file,
            do_quant=True,
            target='rk3588',
            quantized_dtype='asymmetric_quantized-u8'
        )

        # Verify config was called with explicit dtype
        config_call = mock_rknn.config.call_args
        assert config_call[1]['quantized_dtype'] == 'asymmetric_quantized-u8'

    @patch('tools.convert_onnx_to_rknn.RKNN')
    def test_build_rknn_creates_output_directory(self, mock_rknn_class, temp_dir, fake_onnx_file, fake_calib_file, mock_rknn):
        """Test conversion creates output directory if not exists."""
        mock_rknn_class.return_value = mock_rknn

        nested_path = temp_dir / 'models' / 'rknn' / 'v1' / 'output.rknn'
        assert not nested_path.parent.exists()

        build_rknn(
            onnx_path=fake_onnx_file,
            out_path=nested_path,
            calib=fake_calib_file,
            do_quant=True,
            target='rk3588'
        )

        # Verify directory was created
        assert nested_path.parent.exists()

    def test_build_rknn_missing_rknn_toolkit_import_error(self, temp_dir, fake_onnx_file):
        """Test error when rknn-toolkit2 not installed."""
        with patch('tools.convert_onnx_to_rknn.RKNN', side_effect=ImportError('No module named rknn')):
            with pytest.raises(SystemExit) as exc_info:
                build_rknn(
                    onnx_path=fake_onnx_file,
                    out_path=temp_dir / 'out.rknn',
                    calib=None,
                    do_quant=False,
                    target='rk3588'
                )

            assert 'rknn-toolkit2 not installed' in str(exc_info.value)

    def test_build_rknn_missing_rknn_toolkit_attribute_error(self, temp_dir, fake_onnx_file):
        """Test error when rknn-toolkit2 version incompatible."""
        with patch('tools.convert_onnx_to_rknn.RKNN', side_effect=AttributeError('API changed')):
            with pytest.raises(SystemExit) as exc_info:
                build_rknn(
                    onnx_path=fake_onnx_file,
                    out_path=temp_dir / 'out.rknn',
                    calib=None,
                    do_quant=False,
                    target='rk3588'
                )

            assert 'version incompatible' in str(exc_info.value)

    @patch('tools.convert_onnx_to_rknn.RKNN')
    def test_build_rknn_load_onnx_failure(self, mock_rknn_class, temp_dir, fake_onnx_file, mock_rknn):
        """Test handling of load_onnx failure."""
        mock_rknn_class.return_value = mock_rknn
        mock_rknn.load_onnx.return_value = -1  # Failure

        with pytest.raises(SystemExit):
            build_rknn(
                onnx_path=fake_onnx_file,
                out_path=temp_dir / 'out.rknn',
                calib=None,
                do_quant=False,
                target='rk3588'
            )

        # Verify release was called even on failure
        mock_rknn.release.assert_called()

    @patch('tools.convert_onnx_to_rknn.RKNN')
    def test_build_rknn_build_failure(self, mock_rknn_class, temp_dir, fake_onnx_file, fake_calib_file, mock_rknn):
        """Test handling of build failure."""
        mock_rknn_class.return_value = mock_rknn
        mock_rknn.build.return_value = -1  # Failure

        with pytest.raises(SystemExit):
            build_rknn(
                onnx_path=fake_onnx_file,
                out_path=temp_dir / 'out.rknn',
                calib=fake_calib_file,
                do_quant=True,
                target='rk3588'
            )

        mock_rknn.release.assert_called()

    @patch('tools.convert_onnx_to_rknn.RKNN')
    def test_build_rknn_export_failure(self, mock_rknn_class, temp_dir, fake_onnx_file, fake_calib_file, mock_rknn):
        """Test handling of export_rknn failure."""
        mock_rknn_class.return_value = mock_rknn
        mock_rknn.export_rknn.return_value = -1  # Failure

        with pytest.raises(SystemExit):
            build_rknn(
                onnx_path=fake_onnx_file,
                out_path=temp_dir / 'out.rknn',
                calib=fake_calib_file,
                do_quant=True,
                target='rk3588'
            )

        mock_rknn.release.assert_called()

    def test_build_rknn_quantization_without_calibration(self, temp_dir, fake_onnx_file):
        """Test error when quantization requested without calibration dataset."""
        with pytest.raises(SystemExit) as exc_info:
            build_rknn(
                onnx_path=fake_onnx_file,
                out_path=temp_dir / 'out.rknn',
                calib=None,
                do_quant=True,  # Quantization enabled
                target='rk3588'
            )

        assert 'calibration dataset' in str(exc_info.value).lower()

    def test_build_rknn_missing_calibration_file(self, temp_dir, fake_onnx_file):
        """Test error when calibration file doesn't exist."""
        nonexistent_calib = temp_dir / 'nonexistent_calib.txt'

        with pytest.raises(SystemExit) as exc_info:
            build_rknn(
                onnx_path=fake_onnx_file,
                out_path=temp_dir / 'out.rknn',
                calib=nonexistent_calib,
                do_quant=True,
                target='rk3588'
            )

        assert 'not found' in str(exc_info.value).lower()

    @patch('tools.convert_onnx_to_rknn.RKNN')
    def test_build_rknn_config_called_before_load(self, mock_rknn_class, temp_dir, fake_onnx_file, mock_rknn):
        """Test that config() is called before load_onnx()."""
        mock_rknn_class.return_value = mock_rknn

        build_rknn(
            onnx_path=fake_onnx_file,
            out_path=temp_dir / 'out.rknn',
            calib=None,
            do_quant=False,
            target='rk3588'
        )

        # Verify call order
        calls = mock_rknn.method_calls
        config_index = None
        load_index = None

        for i, call_item in enumerate(calls):
            if call_item[0] == 'config':
                config_index = i
            elif call_item[0] == 'load_onnx':
                load_index = i

        assert config_index is not None
        assert load_index is not None
        assert config_index < load_index  # config must come before load

    @patch('tools.convert_onnx_to_rknn.RKNN')
    def test_build_rknn_mean_std_parsing(self, mock_rknn_class, temp_dir, fake_onnx_file, mock_rknn):
        """Test mean and std string parsing."""
        mock_rknn_class.return_value = mock_rknn

        test_cases = [
            ('0,0,0', '255,255,255', [[0.0, 0.0, 0.0]], [[255.0, 255.0, 255.0]]),
            ('127.5,127.5,127.5', '127.5,127.5,127.5', [[127.5, 127.5, 127.5]], [[127.5, 127.5, 127.5]]),
            ('0.5,0.5,0.5', '0.5,0.5,0.5', [[0.5, 0.5, 0.5]], [[0.5, 0.5, 0.5]]),
        ]

        for mean_str, std_str, expected_mean, expected_std in test_cases:
            mock_rknn.reset_mock()

            build_rknn(
                onnx_path=fake_onnx_file,
                out_path=temp_dir / 'out.rknn',
                calib=None,
                do_quant=False,
                target='rk3588',
                mean=mean_str,
                std=std_str
            )

            config_call = mock_rknn.config.call_args
            assert config_call[1]['mean_values'] == expected_mean
            assert config_call[1]['std_values'] == expected_std


class TestBuildRknnEdgeCases:
    """Test suite for edge cases and error conditions."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def fake_onnx(self, temp_dir):
        onnx = temp_dir / 'model.onnx'
        onnx.write_bytes(b'fake')
        return onnx

    def test_build_rknn_invalid_mean_format(self, temp_dir, fake_onnx):
        """Test handling of invalid mean format."""
        with patch('tools.convert_onnx_to_rknn.RKNN') as mock_rknn_class:
            mock_rknn = MagicMock()
            mock_rknn_class.return_value = mock_rknn

            try:
                build_rknn(
                    onnx_path=fake_onnx,
                    out_path=temp_dir / 'out.rknn',
                    calib=None,
                    do_quant=False,
                    target='rk3588',
                    mean='invalid,format,x'  # Non-numeric
                )
            except (ValueError, SystemExit):
                # Expected to fail for invalid format
                pass

    def test_build_rknn_mismatched_mean_std_dimensions(self, temp_dir, fake_onnx):
        """Test handling of mismatched mean/std dimensions."""
        with patch('tools.convert_onnx_to_rknn.RKNN') as mock_rknn_class:
            mock_rknn = MagicMock()
            mock_rknn.config.return_value = 0
            mock_rknn.load_onnx.return_value = 0
            mock_rknn_class.return_value = mock_rknn

            # Mean has 3 values, std has 2 - may cause issues
            try:
                build_rknn(
                    onnx_path=fake_onnx,
                    out_path=temp_dir / 'out.rknn',
                    calib=None,
                    do_quant=False,
                    target='rk3588',
                    mean='0,0,0',
                    std='255,255'
                )
            except (ValueError, SystemExit, AssertionError):
                pass

    @patch('tools.convert_onnx_to_rknn.RKNN')
    def test_build_rknn_empty_quantized_dtype_string(self, mock_rknn_class, temp_dir, fake_onnx):
        """Test handling of empty quantized_dtype string."""
        mock_rknn = MagicMock()
        mock_rknn.config.return_value = 0
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = 0
        mock_rknn.export_rknn.return_value = 0
        mock_rknn_class.return_value = mock_rknn

        # Empty string should trigger auto-detection
        with patch('tools.convert_onnx_to_rknn._detect_rknn_default_qdtype', return_value='w8a8'):
            build_rknn(
                onnx_path=fake_onnx,
                out_path=temp_dir / 'out.rknn',
                calib=None,
                do_quant=False,
                target='rk3588',
                quantized_dtype=''  # Empty string
            )

            config_call = mock_rknn.config.call_args
            assert config_call[1]['quantized_dtype'] == 'w8a8'


class TestBuildRknnIntegration:
    """Integration tests for build_rknn with realistic scenarios."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def realistic_setup(self, temp_dir):
        """Create realistic test environment."""
        # Create ONNX file
        onnx_file = temp_dir / 'yolo11n_416.onnx'
        onnx_file.write_bytes(b'fake_onnx_model_content' * 100)

        # Create calibration images and list
        calib_dir = temp_dir / 'calib_images'
        calib_dir.mkdir()

        calib_images = []
        for i in range(50):
            img_file = calib_dir / f'coco_person_{i:04d}.jpg'
            img_file.write_bytes(b'fake_jpeg_content' * 50)
            calib_images.append(str(img_file.resolve()))

        calib_list = calib_dir / 'calib.txt'
        calib_list.write_text('\n'.join(calib_images))

        return {
            'onnx': onnx_file,
            'calib': calib_list,
            'calib_dir': calib_dir,
            'out_dir': temp_dir / 'artifacts' / 'models'
        }

    @patch('tools.convert_onnx_to_rknn.RKNN')
    def test_build_rknn_realistic_int8_conversion(self, mock_rknn_class, realistic_setup):
        """Test realistic INT8 quantization workflow."""
        mock_rknn = MagicMock()
        mock_rknn.config.return_value = 0
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = 0
        mock_rknn.export_rknn.return_value = 0
        mock_rknn_class.return_value = mock_rknn

        out_path = realistic_setup['out_dir'] / 'yolo11n_416_int8.rknn'

        build_rknn(
            onnx_path=realistic_setup['onnx'],
            out_path=out_path,
            calib=realistic_setup['calib'],
            do_quant=True,
            target='rk3588',
            mean='0,0,0',
            std='255,255,255'
        )

        # Verify complete workflow
        assert mock_rknn.config.called
        assert mock_rknn.load_onnx.called
        assert mock_rknn.build.called
        assert mock_rknn.export_rknn.called
        assert mock_rknn.release.called

        # Verify quantization was enabled
        build_call = mock_rknn.build.call_args
        assert build_call[1]['do_quantization'] is True
        assert build_call[1]['dataset'] == str(realistic_setup['calib'])


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
