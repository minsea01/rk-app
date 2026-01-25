#!/usr/bin/env python3
"""Integration tests for ONNX inference pipeline."""
import pytest
import numpy as np
from pathlib import Path
import cv2
import tempfile

from apps.utils.preprocessing import preprocess_onnx, preprocess_from_array_onnx
from apps.yolov8_rknn_infer import decode_predictions
from apps.config import ModelConfig


@pytest.mark.integration
class TestOnnxInferencePipeline:
    """Integration tests for complete ONNX inference workflow."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            cv2.imwrite(f.name, img)
            yield Path(f.name)
            Path(f.name).unlink()

    @pytest.fixture
    def sample_array(self):
        """Create a sample numpy array."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_preprocessing_to_inference_pipeline(self, sample_image):
        """Test full pipeline from image to preprocessed tensor."""
        # Preprocess image
        input_tensor = preprocess_onnx(sample_image, target_size=640)

        # Validate tensor format
        assert input_tensor.shape == (1, 3, 640, 640)
        assert input_tensor.dtype == np.float32

        # Simulate mock inference output (without actual ONNX model)
        # Mock output: (1, N, C) where N=8400 for 640, C=84 for YOLO
        mock_output = np.random.randn(1, 8400, 84).astype(np.float32)

        # Decode predictions
        boxes, confs, cls_ids = decode_predictions(
            mock_output,
            imgsz=640,
            conf_thres=0.5,
            iou_thres=0.45,
            head='dfl'
        )

        # Validate outputs
        assert isinstance(boxes, np.ndarray)
        assert isinstance(confs, np.ndarray)
        assert isinstance(cls_ids, np.ndarray)
        assert len(boxes) == len(confs) == len(cls_ids)

    def test_array_preprocessing_to_inference(self, sample_array):
        """Test pipeline from numpy array to inference."""
        # Preprocess array
        input_tensor = preprocess_from_array_onnx(sample_array, target_size=416)

        # Validate preprocessing
        assert input_tensor.shape == (1, 3, 416, 416)
        assert input_tensor.dtype == np.float32

        # Mock inference
        # N=3549 for 416: (416/8)^2 + (416/16)^2 + (416/32)^2
        mock_output = np.random.randn(1, 3549, 84).astype(np.float32)

        # Decode
        boxes, confs, cls_ids = decode_predictions(
            mock_output,
            imgsz=416,
            conf_thres=0.25,
            iou_thres=0.45,
            head='dfl'
        )

        assert len(boxes) == len(confs)

    def test_multi_size_inference_pipeline(self, sample_image):
        """Test pipeline with different input sizes."""
        for size in [320, 416, 640]:
            # Preprocess
            input_tensor = preprocess_onnx(sample_image, target_size=size)
            assert input_tensor.shape == (1, 3, size, size)

            # Mock inference with size-appropriate output
            expected_detections = (size // 8) ** 2 + (size // 16) ** 2 + (size // 32) ** 2
            mock_output = np.random.randn(1, expected_detections, 84).astype(np.float32)

            # Decode
            boxes, confs, cls_ids = decode_predictions(
                mock_output,
                imgsz=size,
                conf_thres=0.5,
                iou_thres=0.45,
                head='dfl'
            )

            # Should not crash
            assert isinstance(boxes, np.ndarray)

    def test_preprocessing_consistency(self, sample_array):
        """Test that preprocessing is deterministic."""
        # Preprocess same image twice
        tensor1 = preprocess_from_array_onnx(sample_array.copy(), target_size=640)
        tensor2 = preprocess_from_array_onnx(sample_array.copy(), target_size=640)

        # Should be identical
        np.testing.assert_array_equal(tensor1, tensor2)

    def test_batch_processing_simulation(self, sample_array):
        """Test processing multiple images (simulated batch)."""
        images = [sample_array.copy() for _ in range(5)]
        tensors = []

        for img in images:
            tensor = preprocess_from_array_onnx(img, target_size=416)
            tensors.append(tensor)

        # All tensors should have same shape
        assert all(t.shape == (1, 3, 416, 416) for t in tensors)

        # Simulate batch inference
        for tensor in tensors:
            mock_output = np.random.randn(1, 3549, 84).astype(np.float32)  # 3549 for 416
            boxes, confs, cls_ids = decode_predictions(
                mock_output,
                imgsz=416,
                conf_thres=0.25,
                iou_thres=0.45,
                head='dfl'
            )
            assert len(boxes) == len(confs) == len(cls_ids)

    def test_config_integration(self, sample_image):
        """Test pipeline using configuration values."""
        from apps.config import get_detection_config

        config = get_detection_config(size=416)

        # Use config values
        input_tensor = preprocess_onnx(sample_image, target_size=config['size'])

        assert input_tensor.shape == (1, 3, 416, 416)

        # Mock inference
        mock_output = np.random.randn(1, config['max_detections'], 84).astype(np.float32)

        # Decode with config thresholds
        boxes, confs, cls_ids = decode_predictions(
            mock_output,
            imgsz=config['size'],
            conf_thres=config['conf_threshold'],
            iou_thres=config['iou_threshold'],
            head='dfl'
        )

        # Validate
        assert isinstance(boxes, np.ndarray)

    def test_error_propagation(self):
        """Test that errors propagate correctly through pipeline."""
        from apps.exceptions import PreprocessError

        # Test invalid image path
        with pytest.raises(PreprocessError):
            preprocess_onnx('/nonexistent/image.jpg', target_size=640)


@pytest.mark.integration
@pytest.mark.requires_model
class TestOnnxModelInference:
    """Integration tests requiring actual ONNX model files."""

    def test_real_onnx_inference(self):
        """Test with real ONNX model if available."""
        model_path = Path('artifacts/models/best.onnx')

        if not model_path.exists():
            pytest.skip('ONNX model not found, skipping real inference test')

        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip('onnxruntime not installed')

        # Create test image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            cv2.imwrite(f.name, img)
            img_path = Path(f.name)

        try:
            # Preprocess
            input_size = 640
            input_tensor = preprocess_onnx(img_path, target_size=input_size)

            # Run inference
            session = ort.InferenceSession(str(model_path))
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: input_tensor})

            # Decode predictions
            boxes, confs, cls_ids = decode_predictions(
                outputs[0],
                imgsz=input_size,
                conf_thres=0.25,
                iou_thres=0.45,
                head='dfl'
            )

            # Validate outputs
            assert isinstance(boxes, np.ndarray)
            assert isinstance(confs, np.ndarray)
            assert isinstance(cls_ids, np.ndarray)

        finally:
            img_path.unlink()
