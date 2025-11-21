#!/usr/bin/env python3
"""Unit tests for tools.export_yolov8_to_onnx module.

Tests YOLOv8/YOLO11 model export to ONNX format.
"""
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tools.export_yolov8_to_onnx import export, main
from apps.exceptions import ConfigurationError, ModelLoadError


class TestExportYOLOv8ToONNX:
    """Test suite for YOLO to ONNX export function."""

    def setup_method(self):
        """Create temporary directories for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def test_validates_weights_file_exists(self):
        """Test that ModelLoadError is raised when weights file doesn't exist."""
        non_existent_weights = self.temp_path / 'nonexistent.pt'
        outdir = self.temp_path / 'output'

        with pytest.raises(ModelLoadError, match="Weights file not found"):
            export(
                weights=str(non_existent_weights),
                imgsz=640,
                opset=12,
                simplify=True,
                dynamic=False,
                half=False,
                outdir=outdir
            )

    def test_creates_output_directory(self):
        """Test that output directory is created if it doesn't exist."""
        weights_path = self.temp_path / 'model.pt'
        weights_path.write_text('fake weights')
        outdir = self.temp_path / 'new_output_dir'

        # Mock YOLO class
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # Mock export to return a path
        exported_onnx = self.temp_path / 'model.onnx'
        exported_onnx.write_text('fake onnx')
        mock_model.export.return_value = str(exported_onnx)

        with patch('tools.export_yolov8_to_onnx.YOLO', mock_yolo_class):
            export(
                weights=str(weights_path),
                imgsz=640,
                opset=12,
                simplify=True,
                dynamic=False,
                half=False,
                outdir=outdir
            )

            # Output directory should be created
            assert outdir.exists()
            assert outdir.is_dir()

    def test_raises_configuration_error_when_ultralytics_not_installed(self):
        """Test that ConfigurationError is raised when ultralytics is missing."""
        weights_path = self.temp_path / 'model.pt'
        weights_path.write_text('fake weights')
        outdir = self.temp_path / 'output'

        with patch('tools.export_yolov8_to_onnx.YOLO', side_effect=ImportError("No module named 'ultralytics'")):
            with pytest.raises(ConfigurationError, match="Ultralytics not installed"):
                export(
                    weights=str(weights_path),
                    imgsz=640,
                    opset=12,
                    simplify=True,
                    dynamic=False,
                    half=False,
                    outdir=outdir
                )

    def test_raises_model_load_error_on_invalid_weights(self):
        """Test that ModelLoadError is raised for invalid weights file."""
        weights_path = self.temp_path / 'invalid.pt'
        weights_path.write_text('not a valid model file')
        outdir = self.temp_path / 'output'

        # Mock YOLO class to raise exception on invalid model
        mock_yolo_class = MagicMock()
        mock_yolo_class.side_effect = Exception("Invalid model format")

        with patch('tools.export_yolov8_to_onnx.YOLO', mock_yolo_class):
            with pytest.raises(ModelLoadError, match="Failed to load model"):
                export(
                    weights=str(weights_path),
                    imgsz=640,
                    opset=12,
                    simplify=True,
                    dynamic=False,
                    half=False,
                    outdir=outdir
                )

    def test_calls_model_export_with_correct_parameters(self):
        """Test that model.export() is called with correct parameters."""
        weights_path = self.temp_path / 'model.pt'
        weights_path.write_text('fake weights')
        outdir = self.temp_path / 'output'

        # Mock YOLO class
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # Mock export to return a path
        exported_onnx = outdir / 'model.onnx'
        outdir.mkdir(parents=True, exist_ok=True)
        exported_onnx.write_text('fake onnx')
        mock_model.export.return_value = str(exported_onnx)

        with patch('tools.export_yolov8_to_onnx.YOLO', mock_yolo_class):
            export(
                weights=str(weights_path),
                imgsz=416,
                opset=11,
                simplify=False,
                dynamic=True,
                half=True,
                outdir=outdir
            )

            # Verify export was called with correct parameters
            mock_model.export.assert_called_once_with(
                format='onnx',
                imgsz=416,
                opset=11,
                simplify=False,
                dynamic=True,
                half=True
            )

    def test_moves_onnx_to_target_directory(self):
        """Test that ONNX file is moved to target directory."""
        weights_path = self.temp_path / 'model.pt'
        weights_path.write_text('fake weights')
        outdir = self.temp_path / 'output'

        # Mock YOLO class
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # Mock export to return a path in CWD (different from outdir)
        exported_onnx = self.temp_path / 'model.onnx'
        exported_onnx.write_text('fake onnx content')
        mock_model.export.return_value = str(exported_onnx)

        with patch('tools.export_yolov8_to_onnx.YOLO', mock_yolo_class):
            result = export(
                weights=str(weights_path),
                imgsz=640,
                opset=12,
                simplify=True,
                dynamic=False,
                half=False,
                outdir=outdir
            )

            # Result should be in output directory
            assert result.parent == outdir
            assert result.exists()
            # Content should be copied
            assert result.read_text() == 'fake onnx content'

    def test_uses_custom_output_filename_when_specified(self):
        """Test that custom output filename is used."""
        weights_path = self.temp_path / 'model.pt'
        weights_path.write_text('fake weights')
        outdir = self.temp_path / 'output'
        custom_filename = 'custom_model.onnx'

        # Mock YOLO class
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # Mock export
        exported_onnx = self.temp_path / 'model.onnx'
        exported_onnx.write_text('fake onnx')
        mock_model.export.return_value = str(exported_onnx)

        with patch('tools.export_yolov8_to_onnx.YOLO', mock_yolo_class):
            result = export(
                weights=str(weights_path),
                imgsz=640,
                opset=12,
                simplify=True,
                dynamic=False,
                half=False,
                outdir=outdir,
                outfile=custom_filename
            )

            # Result should have custom filename
            assert result.name == custom_filename

    def test_handles_export_failure_gracefully(self):
        """Test that ModelLoadError is raised when export fails."""
        weights_path = self.temp_path / 'model.pt'
        weights_path.write_text('fake weights')
        outdir = self.temp_path / 'output'

        # Mock YOLO class
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # Mock export to raise exception
        mock_model.export.side_effect = Exception("Export failed due to ONNX error")

        with patch('tools.export_yolov8_to_onnx.YOLO', mock_yolo_class):
            with pytest.raises(ModelLoadError, match="Failed to export model to ONNX"):
                export(
                    weights=str(weights_path),
                    imgsz=640,
                    opset=12,
                    simplify=True,
                    dynamic=False,
                    half=False,
                    outdir=outdir
                )

    def test_handles_file_move_failure(self):
        """Test that ModelLoadError is raised when file move fails."""
        weights_path = self.temp_path / 'model.pt'
        weights_path.write_text('fake weights')
        outdir = self.temp_path / 'output'

        # Mock YOLO class
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # Mock export to return a path
        exported_onnx = self.temp_path / 'model.onnx'
        mock_model.export.return_value = str(exported_onnx)

        with patch('tools.export_yolov8_to_onnx.YOLO', mock_yolo_class):
            # Mock Path.read_bytes to raise exception
            with patch('pathlib.Path.read_bytes', side_effect=Exception("Permission denied")):
                with pytest.raises(ModelLoadError, match="Failed to move ONNX file"):
                    export(
                        weights=str(weights_path),
                        imgsz=640,
                        opset=12,
                        simplify=True,
                        dynamic=False,
                        half=False,
                        outdir=outdir
                    )

    def test_returns_target_path(self):
        """Test that function returns Path to exported ONNX file."""
        weights_path = self.temp_path / 'model.pt'
        weights_path.write_text('fake weights')
        outdir = self.temp_path / 'output'

        # Mock YOLO class
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # Mock export
        exported_onnx = outdir / 'model.onnx'
        outdir.mkdir(parents=True, exist_ok=True)
        exported_onnx.write_text('fake onnx')
        mock_model.export.return_value = str(exported_onnx)

        with patch('tools.export_yolov8_to_onnx.YOLO', mock_yolo_class):
            result = export(
                weights=str(weights_path),
                imgsz=640,
                opset=12,
                simplify=True,
                dynamic=False,
                half=False,
                outdir=outdir
            )

            assert isinstance(result, Path)
            assert result.exists()
            assert result.suffix == '.onnx'


class TestMainFunction:
    """Test suite for main entry point."""

    def test_main_returns_zero_on_success(self):
        """Test that main() returns 0 on successful export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / 'model.pt'
            weights_path.write_text('fake weights')

            # Mock export function
            with patch('tools.export_yolov8_to_onnx.export', return_value=Path(tmpdir) / 'model.onnx'):
                # Mock argparse
                mock_args = [
                    '--weights', str(weights_path),
                    '--imgsz', '640',
                    '--outdir', tmpdir
                ]
                with patch('sys.argv', ['export_yolov8_to_onnx.py'] + mock_args):
                    result = main()
                    assert result == 0

    def test_main_returns_one_on_model_load_error(self):
        """Test that main() returns 1 on ModelLoadError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / 'model.pt'
            weights_path.write_text('fake weights')

            # Mock export to raise ModelLoadError
            with patch('tools.export_yolov8_to_onnx.export', side_effect=ModelLoadError("Load failed")):
                mock_args = [
                    '--weights', str(weights_path),
                    '--imgsz', '640',
                    '--outdir', tmpdir
                ]
                with patch('sys.argv', ['export_yolov8_to_onnx.py'] + mock_args):
                    result = main()
                    assert result == 1

    def test_main_returns_one_on_configuration_error(self):
        """Test that main() returns 1 on ConfigurationError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / 'model.pt'
            weights_path.write_text('fake weights')

            # Mock export to raise ConfigurationError
            with patch('tools.export_yolov8_to_onnx.export', side_effect=ConfigurationError("Config error")):
                mock_args = [
                    '--weights', str(weights_path),
                    '--imgsz', '640',
                    '--outdir', tmpdir
                ]
                with patch('sys.argv', ['export_yolov8_to_onnx.py'] + mock_args):
                    result = main()
                    assert result == 1

    def test_main_returns_one_on_unexpected_error(self):
        """Test that main() returns 1 on unexpected exceptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / 'model.pt'
            weights_path.write_text('fake weights')

            # Mock export to raise unexpected exception
            with patch('tools.export_yolov8_to_onnx.export', side_effect=Exception("Unexpected error")):
                mock_args = [
                    '--weights', str(weights_path),
                    '--imgsz', '640',
                    '--outdir', tmpdir
                ]
                with patch('sys.argv', ['export_yolov8_to_onnx.py'] + mock_args):
                    result = main()
                    assert result == 1
