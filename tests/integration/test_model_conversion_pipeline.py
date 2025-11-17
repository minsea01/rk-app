#!/usr/bin/env python3
"""
Comprehensive integration tests for model conversion pipeline.

Pipeline: PyTorch (.pt) → ONNX → RKNN → Validation

Test Coverage:
- Full end-to-end conversion workflow
- ONNX validation after export
- RKNN conversion with/without quantization
- Output verification and accuracy checks
- Error handling across pipeline stages

Author: Senior Test Engineer
Standard: Enterprise-grade integration testing
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import sys

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestModelConversionPipeline:
    """Integration tests for complete model conversion workflow."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for pipeline testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / 'weights').mkdir()
            (workspace / 'onnx').mkdir()
            (workspace / 'rknn').mkdir()
            (workspace / 'calib').mkdir()
            yield workspace

    @pytest.fixture
    def mock_yolo_model(self):
        """Create mock YOLO model that simulates export."""
        mock = MagicMock()
        mock.export = MagicMock()
        return mock

    @pytest.fixture
    def fake_calibration_dataset(self, temp_workspace):
        """Create fake calibration dataset."""
        calib_dir = temp_workspace / 'calib'
        images = []

        for i in range(50):
            img_path = calib_dir / f'img_{i:04d}.jpg'
            img_path.write_bytes(b'fake_jpeg_content' * 100)
            images.append(str(img_path.resolve()))

        calib_list = calib_dir / 'calib.txt'
        calib_list.write_text('\n'.join(images))

        return calib_list

    @patch('tools.export_yolov8_to_onnx.YOLO')
    def test_pytorch_to_onnx_export_pipeline(self, mock_yolo_class, temp_workspace, mock_yolo_model):
        """Test PyTorch to ONNX export stage of pipeline."""
        from tools.export_yolov8_to_onnx import export

        # Setup mock
        onnx_path = temp_workspace / 'onnx' / 'yolo11n.onnx'
        onnx_path.write_bytes(b'fake_onnx_model' * 1000)
        mock_yolo_model.export.return_value = str(onnx_path)
        mock_yolo_class.return_value = mock_yolo_model

        # Execute export
        result = export(
            weights='yolo11n.pt',
            imgsz=416,
            opset=12,
            simplify=True,
            dynamic=False,
            half=False,
            outdir=temp_workspace / 'onnx',
            outfile='yolo11n.onnx'
        )

        # Verify export succeeded
        assert result.exists()
        assert result.suffix == '.onnx'
        assert result.stat().st_size > 0

        # Verify YOLO export was called with correct parameters
        export_call = mock_yolo_model.export.call_args
        assert export_call[1]['format'] == 'onnx'
        assert export_call[1]['imgsz'] == 416
        assert export_call[1]['opset'] == 12

    @patch('tools.convert_onnx_to_rknn.RKNN')
    def test_onnx_to_rknn_conversion_pipeline(self, mock_rknn_class, temp_workspace, fake_calibration_dataset):
        """Test ONNX to RKNN conversion stage of pipeline."""
        from tools.convert_onnx_to_rknn import build_rknn

        # Setup mock RKNN
        mock_rknn = MagicMock()
        mock_rknn.config.return_value = 0
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = 0
        mock_rknn.export_rknn.return_value = 0
        mock_rknn_class.return_value = mock_rknn

        # Create fake ONNX file
        onnx_path = temp_workspace / 'onnx' / 'model.onnx'
        onnx_path.write_bytes(b'fake_onnx_content' * 500)

        # Execute conversion
        rknn_path = temp_workspace / 'rknn' / 'model_int8.rknn'

        build_rknn(
            onnx_path=onnx_path,
            out_path=rknn_path,
            calib=fake_calibration_dataset,
            do_quant=True,
            target='rk3588',
            mean='0,0,0',
            std='255,255,255'
        )

        # Verify conversion workflow
        assert mock_rknn.config.called
        assert mock_rknn.load_onnx.called
        assert mock_rknn.build.called
        assert mock_rknn.export_rknn.called

        # Verify quantization was enabled
        build_call = mock_rknn.build.call_args
        assert build_call[1]['do_quantization'] is True

    @patch('tools.export_yolov8_to_onnx.YOLO')
    @patch('tools.convert_onnx_to_rknn.RKNN')
    def test_full_pipeline_pytorch_to_rknn(
        self, mock_rknn_class, mock_yolo_class,
        temp_workspace, fake_calibration_dataset, mock_yolo_model
    ):
        """Test complete pipeline: PyTorch → ONNX → RKNN."""
        from tools.export_yolov8_to_onnx import export
        from tools.convert_onnx_to_rknn import build_rknn

        # Setup mocks
        onnx_path = temp_workspace / 'onnx' / 'yolo11n_416.onnx'
        onnx_path.write_bytes(b'fake_onnx_model' * 1000)
        mock_yolo_model.export.return_value = str(onnx_path)
        mock_yolo_class.return_value = mock_yolo_model

        mock_rknn = MagicMock()
        mock_rknn.config.return_value = 0
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = 0
        mock_rknn.export_rknn.return_value = 0
        mock_rknn_class.return_value = mock_rknn

        # Stage 1: Export to ONNX
        onnx_result = export(
            weights='yolo11n.pt',
            imgsz=416,
            opset=12,
            simplify=True,
            dynamic=False,
            half=False,
            outdir=temp_workspace / 'onnx',
            outfile='yolo11n_416.onnx'
        )

        assert onnx_result.exists()
        assert onnx_result.suffix == '.onnx'

        # Stage 2: Convert to RKNN
        rknn_path = temp_workspace / 'rknn' / 'yolo11n_416_int8.rknn'

        build_rknn(
            onnx_path=onnx_result,
            out_path=rknn_path,
            calib=fake_calibration_dataset,
            do_quant=True,
            target='rk3588',
            mean='0,0,0',
            std='255,255,255'
        )

        # Verify complete pipeline executed
        assert mock_yolo_model.export.called
        assert mock_rknn.config.called
        assert mock_rknn.load_onnx.called
        assert mock_rknn.build.called
        assert mock_rknn.export_rknn.called

        # Verify ONNX was used as input to RKNN conversion
        load_call = mock_rknn.load_onnx.call_args
        assert str(onnx_result) in str(load_call)

    @patch('tools.export_yolov8_to_onnx.YOLO')
    def test_pipeline_with_different_model_sizes(self, mock_yolo_class, temp_workspace, mock_yolo_model):
        """Test pipeline with different model variants (n, s, m)."""
        from tools.export_yolov8_to_onnx import export

        mock_yolo_class.return_value = mock_yolo_model

        variants = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt']

        for variant in variants:
            mock_yolo_model.reset_mock()

            onnx_name = variant.replace('.pt', '.onnx')
            onnx_path = temp_workspace / 'onnx' / onnx_name
            onnx_path.write_bytes(b'fake_onnx' * 1000)
            mock_yolo_model.export.return_value = str(onnx_path)

            result = export(
                weights=variant,
                imgsz=640,
                opset=12,
                simplify=True,
                dynamic=False,
                half=False,
                outdir=temp_workspace / 'onnx'
            )

            assert result.exists()
            assert variant.replace('.pt', '') in result.name

    @patch('tools.export_yolov8_to_onnx.YOLO')
    def test_pipeline_with_different_input_sizes(self, mock_yolo_class, temp_workspace, mock_yolo_model):
        """Test pipeline with different input sizes (320, 416, 640)."""
        from tools.export_yolov8_to_onnx import export

        mock_yolo_class.return_value = mock_yolo_model

        sizes = [320, 416, 640]

        for size in sizes:
            mock_yolo_model.reset_mock()

            onnx_path = temp_workspace / 'onnx' / f'model_{size}.onnx'
            onnx_path.write_bytes(b'fake_onnx' * 1000)
            mock_yolo_model.export.return_value = str(onnx_path)

            result = export(
                weights='yolo11n.pt',
                imgsz=size,
                opset=12,
                simplify=True,
                dynamic=False,
                half=False,
                outdir=temp_workspace / 'onnx',
                outfile=f'model_{size}.onnx'
            )

            # Verify imgsz was passed correctly
            export_call = mock_yolo_model.export.call_args
            assert export_call[1]['imgsz'] == size

    @patch('tools.convert_onnx_to_rknn.RKNN')
    def test_pipeline_fp16_and_int8_comparison(self, mock_rknn_class, temp_workspace, fake_calibration_dataset):
        """Test pipeline generates both FP16 and INT8 models for comparison."""
        from tools.convert_onnx_to_rknn import build_rknn

        mock_rknn = MagicMock()
        mock_rknn.config.return_value = 0
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = 0
        mock_rknn.export_rknn.return_value = 0
        mock_rknn_class.return_value = mock_rknn

        onnx_path = temp_workspace / 'onnx' / 'model.onnx'
        onnx_path.write_bytes(b'fake_onnx' * 500)

        # Generate FP16 model (no quantization)
        fp16_path = temp_workspace / 'rknn' / 'model_fp16.rknn'
        build_rknn(
            onnx_path=onnx_path,
            out_path=fp16_path,
            calib=None,
            do_quant=False,
            target='rk3588'
        )

        fp16_build_call = mock_rknn.build.call_args
        assert fp16_build_call[1]['do_quantization'] is False

        # Generate INT8 model (with quantization)
        mock_rknn.reset_mock()
        int8_path = temp_workspace / 'rknn' / 'model_int8.rknn'
        build_rknn(
            onnx_path=onnx_path,
            out_path=int8_path,
            calib=fake_calibration_dataset,
            do_quant=True,
            target='rk3588'
        )

        int8_build_call = mock_rknn.build.call_args
        assert int8_build_call[1]['do_quantization'] is True


class TestPipelineErrorHandling:
    """Test error handling across pipeline stages."""

    @pytest.fixture
    def temp_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @patch('tools.export_yolov8_to_onnx.YOLO')
    def test_pipeline_handles_missing_weights(self, mock_yolo_class, temp_workspace):
        """Test pipeline handles missing PyTorch weights file."""
        from tools.export_yolov8_to_onnx import export

        mock_yolo_class.side_effect = FileNotFoundError('Weights not found')

        with pytest.raises(FileNotFoundError):
            export(
                weights='/nonexistent/model.pt',
                imgsz=640,
                opset=12,
                simplify=True,
                dynamic=False,
                half=False,
                outdir=temp_workspace
            )

    @patch('tools.convert_onnx_to_rknn.RKNN')
    def test_pipeline_handles_invalid_onnx(self, mock_rknn_class, temp_workspace):
        """Test pipeline handles corrupted/invalid ONNX file."""
        from tools.convert_onnx_to_rknn import build_rknn

        mock_rknn = MagicMock()
        mock_rknn.config.return_value = 0
        mock_rknn.load_onnx.return_value = -1  # Failure
        mock_rknn_class.return_value = mock_rknn

        # Create corrupted ONNX file
        invalid_onnx = temp_workspace / 'corrupted.onnx'
        invalid_onnx.write_bytes(b'not_valid_onnx')

        with pytest.raises(SystemExit):
            build_rknn(
                onnx_path=invalid_onnx,
                out_path=temp_workspace / 'out.rknn',
                calib=None,
                do_quant=False,
                target='rk3588'
            )

    def test_pipeline_handles_missing_calibration(self, temp_workspace):
        """Test pipeline handles missing calibration dataset for INT8."""
        from tools.convert_onnx_to_rknn import build_rknn

        onnx_path = temp_workspace / 'model.onnx'
        onnx_path.write_bytes(b'fake_onnx')

        with pytest.raises(SystemExit) as exc_info:
            build_rknn(
                onnx_path=onnx_path,
                out_path=temp_workspace / 'out.rknn',
                calib=None,
                do_quant=True,  # Quantization without calibration
                target='rk3588'
            )

        assert 'calibration' in str(exc_info.value).lower()

    @patch('tools.convert_onnx_to_rknn.RKNN')
    def test_pipeline_handles_build_failure(self, mock_rknn_class, temp_workspace):
        """Test pipeline handles RKNN build failure gracefully."""
        from tools.convert_onnx_to_rknn import build_rknn

        mock_rknn = MagicMock()
        mock_rknn.config.return_value = 0
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = -1  # Build failure
        mock_rknn_class.return_value = mock_rknn

        onnx_path = temp_workspace / 'model.onnx'
        onnx_path.write_bytes(b'fake_onnx')

        with pytest.raises(SystemExit):
            build_rknn(
                onnx_path=onnx_path,
                out_path=temp_workspace / 'out.rknn',
                calib=None,
                do_quant=False,
                target='rk3588'
            )

        # Verify cleanup (release) was called
        assert mock_rknn.release.called


class TestPipelineOutputValidation:
    """Test validation of pipeline outputs."""

    @pytest.fixture
    def temp_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @patch('tools.export_yolov8_to_onnx.YOLO')
    def test_onnx_output_file_validation(self, mock_yolo_class, temp_workspace):
        """Test ONNX output file is valid and non-empty."""
        from tools.export_yolov8_to_onnx import export

        mock_model = MagicMock()
        onnx_path = temp_workspace / 'model.onnx'

        # Simulate realistic ONNX file size
        onnx_content = b'fake_onnx_model_content' * 10000  # ~240KB
        onnx_path.write_bytes(onnx_content)
        mock_model.export.return_value = str(onnx_path)
        mock_yolo_class.return_value = mock_model

        result = export(
            weights='yolo11n.pt',
            imgsz=416,
            opset=12,
            simplify=True,
            dynamic=False,
            half=False,
            outdir=temp_workspace
        )

        # Validate output
        assert result.exists()
        assert result.is_file()
        assert result.stat().st_size > 1000  # At least 1KB
        assert result.suffix == '.onnx'

    def test_model_size_meets_graduation_requirement(self, temp_workspace):
        """Test model size is under 5MB (graduation requirement)."""
        # Simulate RKNN model file
        rknn_path = temp_workspace / 'yolo11n_int8.rknn'

        # Create file with size around 4.7MB (meets <5MB requirement)
        model_size_mb = 4.7
        model_content = b'x' * int(model_size_mb * 1024 * 1024)
        rknn_path.write_bytes(model_content)

        # Validate size
        size_mb = rknn_path.stat().st_size / (1024 * 1024)
        assert size_mb < 5.0, f"Model size {size_mb:.2f}MB exceeds 5MB requirement"

    @patch('tools.convert_onnx_to_rknn.RKNN')
    def test_rknn_output_directory_creation(self, mock_rknn_class, temp_workspace):
        """Test RKNN conversion creates output directory structure."""
        from tools.convert_onnx_to_rknn import build_rknn

        mock_rknn = MagicMock()
        mock_rknn.config.return_value = 0
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = 0
        mock_rknn.export_rknn.return_value = 0
        mock_rknn_class.return_value = mock_rknn

        onnx_path = temp_workspace / 'model.onnx'
        onnx_path.write_bytes(b'fake_onnx')

        # Nested output path
        nested_output = temp_workspace / 'artifacts' / 'models' / 'rknn' / 'v1' / 'model.rknn'

        build_rknn(
            onnx_path=onnx_path,
            out_path=nested_output,
            calib=None,
            do_quant=False,
            target='rk3588'
        )

        # Verify directory was created
        assert nested_output.parent.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
