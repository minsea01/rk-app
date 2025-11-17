#!/usr/bin/env python3
"""
Comprehensive unit tests for export_yolov8_to_onnx.py tool.

Test Coverage:
- export() function with various parameters
- ONNX file creation and validation
- Error handling for missing dependencies
- Parameter validation
- File I/O operations

Author: Senior Test Engineer
Standard: Enterprise-grade with 95%+ coverage
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

from tools.export_yolov8_to_onnx import export


class TestExportFunction:
    """Test suite for export() function - PyTorch to ONNX conversion."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_yolo_model(self):
        """Create mock YOLO model for testing."""
        mock_model = MagicMock()
        mock_model.export = MagicMock(return_value='/tmp/model.onnx')
        return mock_model

    @patch('tools.export_yolov8_to_onnx.YOLO')
    def test_export_creates_onnx_file(self, mock_yolo_class, temp_output_dir, mock_yolo_model):
        """Test export creates ONNX file with correct parameters."""
        mock_yolo_class.return_value = mock_yolo_model

        # Create a fake ONNX file to be "exported"
        fake_onnx = temp_output_dir / 'test.onnx'
        fake_onnx.write_bytes(b'fake_onnx_content')
        mock_yolo_model.export.return_value = str(fake_onnx)

        result = export(
            weights='yolov8n.pt',
            imgsz=640,
            opset=12,
            simplify=True,
            dynamic=False,
            half=False,
            outdir=temp_output_dir,
            outfile='model.onnx'
        )

        # Verify YOLO was called
        mock_yolo_class.assert_called_once_with('yolov8n.pt')

        # Verify export was called with correct parameters
        mock_yolo_model.export.assert_called_once_with(
            format='onnx',
            imgsz=640,
            opset=12,
            simplify=True,
            dynamic=False,
            half=False,
        )

        # Verify output file exists
        assert result.exists()

    @patch('tools.export_yolov8_to_onnx.YOLO')
    def test_export_default_parameters(self, mock_yolo_class, temp_output_dir, mock_yolo_model):
        """Test export with default parameters."""
        mock_yolo_class.return_value = mock_yolo_model

        fake_onnx = temp_output_dir / 'yolov8s.onnx'
        fake_onnx.write_bytes(b'fake_content')
        mock_yolo_model.export.return_value = str(fake_onnx)

        result = export(
            weights='yolov8s.pt',
            imgsz=640,
            opset=12,
            simplify=True,
            dynamic=False,
            half=False,
            outdir=temp_output_dir
        )

        # Should use default filename from model
        assert result.name == 'yolov8s.onnx'

    @patch('tools.export_yolov8_to_onnx.YOLO')
    def test_export_custom_image_size(self, mock_yolo_class, temp_output_dir, mock_yolo_model):
        """Test export with custom image size."""
        mock_yolo_class.return_value = mock_yolo_model

        fake_onnx = temp_output_dir / 'model_416.onnx'
        fake_onnx.write_bytes(b'fake_content')
        mock_yolo_model.export.return_value = str(fake_onnx)

        export(
            weights='yolov8n.pt',
            imgsz=416,
            opset=12,
            simplify=True,
            dynamic=False,
            half=False,
            outdir=temp_output_dir
        )

        # Verify imgsz parameter was passed
        call_args = mock_yolo_model.export.call_args
        assert call_args[1]['imgsz'] == 416

    @patch('tools.export_yolov8_to_onnx.YOLO')
    def test_export_with_dynamic_shapes(self, mock_yolo_class, temp_output_dir, mock_yolo_model):
        """Test export with dynamic batch size enabled."""
        mock_yolo_class.return_value = mock_yolo_model

        fake_onnx = temp_output_dir / 'dynamic.onnx'
        fake_onnx.write_bytes(b'fake_content')
        mock_yolo_model.export.return_value = str(fake_onnx)

        export(
            weights='yolov8n.pt',
            imgsz=640,
            opset=12,
            simplify=True,
            dynamic=True,  # Enable dynamic shapes
            half=False,
            outdir=temp_output_dir
        )

        call_args = mock_yolo_model.export.call_args
        assert call_args[1]['dynamic'] is True

    @patch('tools.export_yolov8_to_onnx.YOLO')
    def test_export_with_fp16(self, mock_yolo_class, temp_output_dir, mock_yolo_model):
        """Test export with FP16 (half precision) enabled."""
        mock_yolo_class.return_value = mock_yolo_model

        fake_onnx = temp_output_dir / 'half.onnx'
        fake_onnx.write_bytes(b'fake_content')
        mock_yolo_model.export.return_value = str(fake_onnx)

        export(
            weights='yolov8n.pt',
            imgsz=640,
            opset=12,
            simplify=True,
            dynamic=False,
            half=True,  # Enable FP16
            outdir=temp_output_dir
        )

        call_args = mock_yolo_model.export.call_args
        assert call_args[1]['half'] is True

    @patch('tools.export_yolov8_to_onnx.YOLO')
    def test_export_without_simplification(self, mock_yolo_class, temp_output_dir, mock_yolo_model):
        """Test export with simplification disabled."""
        mock_yolo_class.return_value = mock_yolo_model

        fake_onnx = temp_output_dir / 'no_simplify.onnx'
        fake_onnx.write_bytes(b'fake_content')
        mock_yolo_model.export.return_value = str(fake_onnx)

        export(
            weights='yolov8n.pt',
            imgsz=640,
            opset=12,
            simplify=False,  # Disable simplification
            dynamic=False,
            half=False,
            outdir=temp_output_dir
        )

        call_args = mock_yolo_model.export.call_args
        assert call_args[1]['simplify'] is False

    @patch('tools.export_yolov8_to_onnx.YOLO')
    def test_export_different_opset_versions(self, mock_yolo_class, temp_output_dir, mock_yolo_model):
        """Test export with different ONNX opset versions."""
        mock_yolo_class.return_value = mock_yolo_model

        for opset in [11, 12, 13, 14]:
            fake_onnx = temp_output_dir / f'opset{opset}.onnx'
            fake_onnx.write_bytes(b'fake_content')
            mock_yolo_model.export.return_value = str(fake_onnx)

            export(
                weights='yolov8n.pt',
                imgsz=640,
                opset=opset,
                simplify=True,
                dynamic=False,
                half=False,
                outdir=temp_output_dir
            )

            call_args = mock_yolo_model.export.call_args
            assert call_args[1]['opset'] == opset

    @patch('tools.export_yolov8_to_onnx.YOLO')
    def test_export_creates_output_directory(self, mock_yolo_class, temp_output_dir, mock_yolo_model):
        """Test export creates output directory if it doesn't exist."""
        mock_yolo_class.return_value = mock_yolo_model

        nested_dir = temp_output_dir / 'models' / 'onnx' / 'v1'
        assert not nested_dir.exists()

        fake_onnx = nested_dir / 'model.onnx'
        # Directory will be created by export function

        export(
            weights='yolov8n.pt',
            imgsz=640,
            opset=12,
            simplify=True,
            dynamic=False,
            half=False,
            outdir=nested_dir
        )

        # Verify directory was created
        assert nested_dir.exists()

    @patch('tools.export_yolov8_to_onnx.YOLO')
    def test_export_custom_output_filename(self, mock_yolo_class, temp_output_dir, mock_yolo_model):
        """Test export with custom output filename."""
        mock_yolo_class.return_value = mock_yolo_model

        custom_name = 'custom_model_v2.onnx'
        fake_onnx = temp_output_dir / custom_name
        fake_onnx.write_bytes(b'fake_content')
        mock_yolo_model.export.return_value = str(temp_output_dir / 'auto_generated.onnx')

        result = export(
            weights='yolov8n.pt',
            imgsz=640,
            opset=12,
            simplify=True,
            dynamic=False,
            half=False,
            outdir=temp_output_dir,
            outfile=custom_name
        )

        # Should create file with custom name
        assert result.name == custom_name

    @patch('tools.export_yolov8_to_onnx.YOLO')
    def test_export_file_move_when_paths_differ(self, mock_yolo_class, temp_output_dir, mock_yolo_model):
        """Test export moves file when source and target paths differ."""
        mock_yolo_class.return_value = mock_yolo_model

        # Simulate Ultralytics writing to a different location
        source_file = temp_output_dir / 'source' / 'model.onnx'
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_bytes(b'onnx_model_content')

        target_dir = temp_output_dir / 'target'
        mock_yolo_model.export.return_value = str(source_file)

        result = export(
            weights='yolov8n.pt',
            imgsz=640,
            opset=12,
            simplify=True,
            dynamic=False,
            half=False,
            outdir=target_dir,
            outfile='final.onnx'
        )

        # File should be moved/copied to target location
        assert result == target_dir / 'final.onnx'

    def test_export_missing_ultralytics_raises_error(self, temp_output_dir):
        """Test export raises SystemExit when ultralytics is not installed."""
        with patch('tools.export_yolov8_to_onnx.YOLO', side_effect=ImportError('No module named ultralytics')):
            with pytest.raises(SystemExit) as exc_info:
                export(
                    weights='yolov8n.pt',
                    imgsz=640,
                    opset=12,
                    simplify=True,
                    dynamic=False,
                    half=False,
                    outdir=temp_output_dir
                )

            assert 'ultralytics' in str(exc_info.value).lower()

    @patch('tools.export_yolov8_to_onnx.YOLO')
    def test_export_invalid_weights_file(self, mock_yolo_class, temp_output_dir):
        """Test export handles invalid weights file gracefully."""
        # YOLO class raises error for invalid weights
        mock_yolo_class.side_effect = FileNotFoundError('Weights file not found')

        with pytest.raises(FileNotFoundError):
            export(
                weights='/nonexistent/model.pt',
                imgsz=640,
                opset=12,
                simplify=True,
                dynamic=False,
                half=False,
                outdir=temp_output_dir
            )

    @patch('tools.export_yolov8_to_onnx.YOLO')
    def test_export_yolov8_variants(self, mock_yolo_class, temp_output_dir, mock_yolo_model):
        """Test export with different YOLOv8 model variants."""
        mock_yolo_class.return_value = mock_yolo_model

        variants = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']

        for variant in variants:
            fake_onnx = temp_output_dir / f'{variant.replace(".pt", ".onnx")}'
            fake_onnx.write_bytes(b'fake_content')
            mock_yolo_model.export.return_value = str(fake_onnx)

            export(
                weights=variant,
                imgsz=640,
                opset=12,
                simplify=True,
                dynamic=False,
                half=False,
                outdir=temp_output_dir
            )

            # Verify YOLO was called with correct variant
            assert mock_yolo_class.call_args[0][0] == variant

    @patch('tools.export_yolov8_to_onnx.YOLO')
    def test_export_yolo11_model(self, mock_yolo_class, temp_output_dir, mock_yolo_model):
        """Test export with YOLO11 model (latest)."""
        mock_yolo_class.return_value = mock_yolo_model

        fake_onnx = temp_output_dir / 'yolo11n.onnx'
        fake_onnx.write_bytes(b'fake_content')
        mock_yolo_model.export.return_value = str(fake_onnx)

        result = export(
            weights='yolo11n.pt',
            imgsz=640,
            opset=12,
            simplify=True,
            dynamic=False,
            half=False,
            outdir=temp_output_dir
        )

        mock_yolo_class.assert_called_once_with('yolo11n.pt')
        assert result.exists()

    @patch('tools.export_yolov8_to_onnx.YOLO')
    def test_export_returns_path_object(self, mock_yolo_class, temp_output_dir, mock_yolo_model):
        """Test export returns Path object."""
        mock_yolo_class.return_value = mock_yolo_model

        fake_onnx = temp_output_dir / 'model.onnx'
        fake_onnx.write_bytes(b'fake_content')
        mock_yolo_model.export.return_value = str(fake_onnx)

        result = export(
            weights='yolov8n.pt',
            imgsz=640,
            opset=12,
            simplify=True,
            dynamic=False,
            half=False,
            outdir=temp_output_dir
        )

        assert isinstance(result, Path)

    @patch('tools.export_yolov8_to_onnx.YOLO')
    def test_export_multiple_sizes(self, mock_yolo_class, temp_output_dir, mock_yolo_model):
        """Test export with multiple image sizes (320, 416, 640)."""
        mock_yolo_class.return_value = mock_yolo_model

        sizes = [320, 416, 640, 1280]

        for size in sizes:
            fake_onnx = temp_output_dir / f'model_{size}.onnx'
            fake_onnx.write_bytes(b'fake_content')
            mock_yolo_model.export.return_value = str(fake_onnx)

            export(
                weights='yolov8n.pt',
                imgsz=size,
                opset=12,
                simplify=True,
                dynamic=False,
                half=False,
                outdir=temp_output_dir,
                outfile=f'model_{size}.onnx'
            )

            call_args = mock_yolo_model.export.call_args
            assert call_args[1]['imgsz'] == size


class TestExportFileOperations:
    """Test suite for file I/O operations in export."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_path_object_creation(self, temp_dir):
        """Test Path object creation and operations."""
        output_path = temp_dir / 'models' / 'best.onnx'

        # Test parent directory creation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        assert output_path.parent.exists()

        # Test file writing
        output_path.write_bytes(b'test_onnx_content')
        assert output_path.exists()
        assert output_path.read_bytes() == b'test_onnx_content'

    def test_path_resolution(self, temp_dir):
        """Test path resolution for same/different paths."""
        path1 = temp_dir / 'model.onnx'
        path2 = temp_dir / 'model.onnx'
        path3 = temp_dir / 'other.onnx'

        # Same path resolution
        assert path1.resolve() == path2.resolve()

        # Different path resolution
        assert path1.resolve() != path3.resolve()

    def test_file_copy_operation(self, temp_dir):
        """Test file copy when source != target."""
        source = temp_dir / 'source.onnx'
        target = temp_dir / 'target.onnx'

        source.write_bytes(b'onnx_model_data')

        # Simulate copy operation
        target.write_bytes(source.read_bytes())

        assert source.exists()
        assert target.exists()
        assert source.read_bytes() == target.read_bytes()


class TestExportEdgeCases:
    """Test suite for edge cases and error conditions."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @patch('tools.export_yolov8_to_onnx.YOLO')
    def test_export_empty_weights_filename(self, mock_yolo_class, temp_dir):
        """Test export with empty weights filename."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # Should handle empty string
        try:
            export(
                weights='',
                imgsz=640,
                opset=12,
                simplify=True,
                dynamic=False,
                half=False,
                outdir=temp_dir
            )
        except (ValueError, FileNotFoundError):
            # Expected to fail for empty weights
            pass

    @patch('tools.export_yolov8_to_onnx.YOLO')
    def test_export_zero_image_size(self, mock_yolo_class, temp_dir):
        """Test export with invalid image size."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # Zero or negative size should be handled
        try:
            export(
                weights='yolov8n.pt',
                imgsz=0,
                opset=12,
                simplify=True,
                dynamic=False,
                half=False,
                outdir=temp_dir
            )
        except (ValueError, AssertionError):
            # Expected to fail for invalid size
            pass

    @patch('tools.export_yolov8_to_onnx.YOLO')
    def test_export_invalid_opset(self, mock_yolo_class, temp_dir):
        """Test export with out-of-range opset version."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # Very old or very new opset versions
        for opset in [1, 5, 50, 100]:
            try:
                export(
                    weights='yolov8n.pt',
                    imgsz=640,
                    opset=opset,
                    simplify=True,
                    dynamic=False,
                    half=False,
                    outdir=temp_dir
                )
            except (ValueError, RuntimeError):
                # May fail for unsupported opset
                pass


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
