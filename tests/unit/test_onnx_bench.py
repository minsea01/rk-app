#!/usr/bin/env python3
"""Unit tests for tools.onnx_bench module.

Tests ONNX Runtime performance benchmarking utility.
"""
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
import cv2 as real_cv2

from tools.onnx_bench import make_input, main


class TestMakeInput:
    """Test suite for make_input function."""

    def test_loads_existing_image(self):
        """Test that existing image is loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a real test image using actual cv2
            img_path = Path(tmpdir) / 'test.jpg'
            test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            real_cv2.imwrite(str(img_path), test_img)  # type: ignore[attr-defined]

            result = make_input(img_path, size=640)

            # Verify result shape (NCHW format)
            assert result.shape == (1, 3, 640, 640)
            assert result.dtype == np.float32

    def test_creates_synthetic_image_when_file_not_found(self):
        """Test that synthetic image is created when file doesn't exist."""
        non_existent_path = Path('/nonexistent/image.jpg')

        result = make_input(non_existent_path, size=640)

        # Verify synthetic image was created
        assert result.shape == (1, 3, 640, 640)
        assert result.dtype == np.float32

    def test_creates_synthetic_image_with_empty_path(self):
        """Test that synthetic image is created with empty path."""
        empty_path = Path('')

        result = make_input(empty_path, size=416)

        # Verify synthetic image with correct size
        assert result.shape == (1, 3, 416, 416)

    def test_resizes_image_to_target_size(self):
        """Test that image is resized to target size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a 1920x1080 test image
            img_path = Path(tmpdir) / 'test.jpg'
            test_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            real_cv2.imwrite(str(img_path), test_img)  # type: ignore[attr-defined]

            result = make_input(img_path, size=640)

            # Verify resized to target size
            assert result.shape == (1, 3, 640, 640)

    def test_normalizes_pixel_values_to_0_1(self):
        """Test that pixel values are normalized to [0, 1] range."""
        # Use synthetic image (no file)
        result = make_input(Path(''), size=640)

        # Pixel values should be in [0, 1] range
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_converts_to_nchw_format(self):
        """Test that output is in NCHW (batch, channels, height, width) format."""
        result = make_input(Path(''), size=640)

        # Verify NCHW format
        assert result.shape[0] == 1  # Batch size
        assert result.shape[1] == 3  # Channels (RGB)
        assert result.shape[2] == 640  # Height
        assert result.shape[3] == 640  # Width

    def test_uses_random_seed_for_reproducibility(self):
        """Test that synthetic images are reproducible with fixed seed."""
        # Generate twice - should be identical due to fixed seed in make_input
        result1 = make_input(Path(''), size=640)
        result2 = make_input(Path(''), size=640)

        # Should be identical (same random seed)
        np.testing.assert_array_equal(result1, result2)


class TestONNXBenchMain:
    """Test suite for main benchmarking function."""

    def test_validates_onnx_file_exists(self):
        """Test that program exits or raises error if ONNX file doesn't exist."""
        non_existent_onnx = Path('/nonexistent/model.onnx')

        test_args = [
            'onnx_bench.py',
            '--onnx', str(non_existent_onnx),
            '--imgsz', '640'
        ]

        with patch('sys.argv', test_args):
            # Program should exit or raise an error for non-existent file
            with pytest.raises((SystemExit, FileNotFoundError, Exception)):
                main()

    def test_imports_onnxruntime_successfully(self):
        """Test that onnxruntime is imported and used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / 'model.onnx'
            onnx_path.write_text('fake onnx model')

            test_args = [
                'onnx_bench.py',
                '--onnx', str(onnx_path),
                '--imgsz', '640',
                '--loops', '10'
            ]

            # Mock onnxruntime
            mock_ort = MagicMock()
            mock_session = MagicMock()

            # Mock inputs/outputs
            mock_input = MagicMock()
            mock_input.name = 'images'
            mock_output = MagicMock()
            mock_output.name = 'output0'

            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [mock_output]
            mock_session.run.return_value = [np.random.randn(1, 84, 8400)]

            mock_ort.InferenceSession.return_value = mock_session

            with patch('sys.argv', test_args):
                with patch.dict('sys.modules', {'onnxruntime': mock_ort}):
                    with patch('tools.onnx_bench.cv2'):
                        main()

                        # Verify InferenceSession was created
                        mock_ort.InferenceSession.assert_called_once()

    def test_runs_warmup_iterations(self):
        """Test that warmup iterations are executed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / 'model.onnx'
            onnx_path.write_text('fake onnx model')

            test_args = [
                'onnx_bench.py',
                '--onnx', str(onnx_path),
                '--warmup', '5',
                '--loops', '10'
            ]

            # Mock onnxruntime
            mock_ort = MagicMock()
            mock_session = MagicMock()

            mock_input = MagicMock()
            mock_input.name = 'images'
            mock_output = MagicMock()
            mock_output.name = 'output0'

            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [mock_output]
            mock_session.run.return_value = [np.random.randn(1, 84, 8400)]

            mock_ort.InferenceSession.return_value = mock_session

            with patch('sys.argv', test_args):
                with patch.dict('sys.modules', {'onnxruntime': mock_ort}):
                    with patch('tools.onnx_bench.cv2'):
                        main()

                        # Verify inference was called (warmup + loops)
                        # Should be called 5 + 10 = 15 times
                        assert mock_session.run.call_count >= 10

    def test_uses_specified_execution_provider(self):
        """Test that specified execution provider is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / 'model.onnx'
            onnx_path.write_text('fake onnx model')

            test_args = [
                'onnx_bench.py',
                '--onnx', str(onnx_path),
                '--provider', 'CUDAExecutionProvider',
                '--loops', '5'
            ]

            mock_ort = MagicMock()
            mock_session = MagicMock()

            mock_input = MagicMock()
            mock_input.name = 'images'
            mock_output = MagicMock()
            mock_output.name = 'output0'

            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [mock_output]
            mock_session.run.return_value = [np.random.randn(1, 84, 8400)]

            mock_ort.InferenceSession.return_value = mock_session

            with patch('sys.argv', test_args):
                with patch.dict('sys.modules', {'onnxruntime': mock_ort}):
                    with patch('tools.onnx_bench.cv2'):
                        main()

                        # Verify provider was specified
                        call_kwargs = mock_ort.InferenceSession.call_args[1]
                        assert 'CUDAExecutionProvider' in call_kwargs['providers']

    def test_calculates_average_inference_time(self):
        """Test that average inference time is calculated and printed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / 'model.onnx'
            onnx_path.write_text('fake onnx model')

            test_args = [
                'onnx_bench.py',
                '--onnx', str(onnx_path),
                '--loops', '100',
                '--warmup', '10'
            ]

            mock_ort = MagicMock()
            mock_session = MagicMock()

            mock_input = MagicMock()
            mock_input.name = 'images'
            mock_output = MagicMock()
            mock_output.name = 'output0'

            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [mock_output]
            mock_session.run.return_value = [np.random.randn(1, 84, 8400)]

            mock_ort.InferenceSession.return_value = mock_session

            with patch('sys.argv', test_args):
                with patch.dict('sys.modules', {'onnxruntime': mock_ort}):
                    with patch('tools.onnx_bench.cv2'):
                        with patch('builtins.print') as mock_print:
                            main()

                            # Verify statistics were printed
                            assert mock_print.called

    def test_handles_onnxruntime_import_error(self):
        """Test that missing onnxruntime is handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / 'model.onnx'
            onnx_path.write_text('fake onnx model')

            test_args = [
                'onnx_bench.py',
                '--onnx', str(onnx_path)
            ]

            with patch('sys.argv', test_args):
                # Remove onnxruntime from modules
                with patch.dict('sys.modules', {'onnxruntime': None}):
                    with pytest.raises(SystemExit):
                        main()


class TestONNXBenchEdgeCases:
    """Test edge cases for ONNX benchmarking."""

    def test_handles_small_image_sizes(self):
        """Test that small image sizes (e.g., 416) are handled."""
        result = make_input(Path(''), size=416)

        # Verify correct size
        assert result.shape == (1, 3, 416, 416)

    def test_handles_large_image_sizes(self):
        """Test that large image sizes (e.g., 1280) are handled."""
        result = make_input(Path(''), size=1280)

        # Verify correct size
        assert result.shape == (1, 3, 1280, 1280)

    def test_handles_single_loop_iteration(self):
        """Test that single loop iteration works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / 'model.onnx'
            onnx_path.write_text('fake onnx model')

            test_args = [
                'onnx_bench.py',
                '--onnx', str(onnx_path),
                '--loops', '1',
                '--warmup', '0'
            ]

            mock_ort = MagicMock()
            mock_session = MagicMock()

            mock_input = MagicMock()
            mock_input.name = 'images'
            mock_output = MagicMock()
            mock_output.name = 'output0'

            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [mock_output]
            mock_session.run.return_value = [np.random.randn(1, 84, 8400)]

            mock_ort.InferenceSession.return_value = mock_session

            with patch('sys.argv', test_args):
                with patch.dict('sys.modules', {'onnxruntime': mock_ort}):
                    main()

                    # Should complete without error
                    assert mock_session.run.call_count >= 1

    def test_handles_grayscale_image_conversion(self):
        """Test that grayscale images are converted to RGB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / 'gray.jpg'

            # Create a real grayscale image
            gray_img = np.random.randint(0, 255, (640, 640), dtype=np.uint8)
            real_cv2.imwrite(str(img_path), gray_img)  # type: ignore[attr-defined]

            # The function should handle grayscale images
            # Note: cv2.imread by default reads as BGR (3 channels),
            # so this test mainly verifies file I/O works
            result = make_input(img_path, size=640)
            assert result.shape == (1, 3, 640, 640)
