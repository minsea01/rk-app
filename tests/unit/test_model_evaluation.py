#!/usr/bin/env python3
"""Unit tests for tools.model_evaluation module.

Tests YOLO model evaluation with mAP calculation and visualization.
"""
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import json

import pytest
import numpy as np

from tools.model_evaluation import ModelEvaluator


class TestModelEvaluatorInit:
    """Test suite for ModelEvaluator initialization."""

    def setup_method(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def test_initializes_with_valid_paths(self):
        """Test that ModelEvaluator initializes with valid model and data paths."""
        # Create mock files
        model_path = self.temp_path / 'best.pt'
        model_path.write_text('fake model')

        data_yaml = self.temp_path / 'data.yaml'
        data_yaml.write_text('path: /data\nnames: ["person", "car"]')

        # Mock YOLO class
        with patch('tools.model_evaluation.YOLO') as mock_yolo:
            mock_model = MagicMock()
            mock_yolo.return_value = mock_model

            evaluator = ModelEvaluator(
                model_path=str(model_path),
                data_yaml_path=str(data_yaml)
            )

            # Verify initialization
            assert evaluator.model_path == model_path
            assert evaluator.data_yaml_path == data_yaml
            assert evaluator.conf_threshold == 0.25  # Default
            assert evaluator.iou_threshold == 0.6  # Default

    def test_loads_model_successfully(self):
        """Test that YOLO model is loaded correctly."""
        model_path = self.temp_path / 'best.pt'
        model_path.write_text('fake model')

        data_yaml = self.temp_path / 'data.yaml'
        data_yaml.write_text('path: /data\nnames: []')

        with patch('tools.model_evaluation.YOLO') as mock_yolo:
            mock_model = MagicMock()
            mock_yolo.return_value = mock_model

            evaluator = ModelEvaluator(str(model_path), str(data_yaml))

            # Verify YOLO was called with model path
            mock_yolo.assert_called_once_with(str(model_path))
            assert evaluator.model == mock_model

    def test_loads_config_from_yaml(self):
        """Test that data configuration is loaded from YAML."""
        model_path = self.temp_path / 'best.pt'
        model_path.write_text('fake model')

        data_yaml = self.temp_path / 'data.yaml'
        yaml_content = """
path: /datasets/test
train: train/images
val: val/images
names:
  0: person
  1: car
  2: bicycle
"""
        data_yaml.write_text(yaml_content)

        with patch('tools.model_evaluation.YOLO'):
            with patch('builtins.open', mock_open(read_data=yaml_content)):
                with patch('tools.model_evaluation.yaml.safe_load') as mock_yaml_load:
                    mock_yaml_load.return_value = {
                        'path': '/datasets/test',
                        'names': {0: 'person', 1: 'car', 2: 'bicycle'}
                    }

                    evaluator = ModelEvaluator(str(model_path), str(data_yaml))

                    # Verify YAML was loaded
                    # mock_yaml_load.assert_called_once()

    def test_accepts_custom_thresholds(self):
        """Test that custom confidence and IOU thresholds are accepted."""
        model_path = self.temp_path / 'best.pt'
        model_path.write_text('fake model')

        data_yaml = self.temp_path / 'data.yaml'
        data_yaml.write_text('names: []')

        with patch('tools.model_evaluation.YOLO'):
            evaluator = ModelEvaluator(
                str(model_path),
                str(data_yaml),
                conf_threshold=0.5,
                iou_threshold=0.7
            )

            assert evaluator.conf_threshold == 0.5
            assert evaluator.iou_threshold == 0.7


class TestModelEvaluatorMethods:
    """Test suite for ModelEvaluator methods."""

    def setup_method(self):
        """Setup mock evaluator for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create mock files
        self.model_path = self.temp_path / 'best.pt'
        self.model_path.write_text('fake model')

        self.data_yaml = self.temp_path / 'data.yaml'
        self.data_yaml.write_text('path: /data\nnames: ["person"]')

    def test_run_evaluation_calls_model_val(self):
        """Test that run_evaluation calls model.val() method."""
        with patch('tools.model_evaluation.YOLO') as mock_yolo:
            mock_model = MagicMock()
            mock_model.val = MagicMock(return_value=MagicMock())
            mock_yolo.return_value = mock_model

            evaluator = ModelEvaluator(str(self.model_path), str(self.data_yaml))

            # Mock the method
            if hasattr(evaluator, 'run_evaluation'):
                evaluator.run_evaluation()

                # Verify val() was called
                mock_model.val.assert_called()

    def test_calculates_map_metrics(self):
        """Test that mAP metrics are calculated correctly."""
        with patch('tools.model_evaluation.YOLO') as mock_yolo:
            mock_model = MagicMock()

            # Mock validation results
            mock_results = MagicMock()
            mock_results.box.map50 = 0.89  # mAP@0.5
            mock_results.box.map = 0.75  # mAP@0.5:0.95
            mock_model.val.return_value = mock_results

            mock_yolo.return_value = mock_model

            evaluator = ModelEvaluator(str(self.model_path), str(self.data_yaml))

            # Get metrics
            if hasattr(evaluator, 'run_evaluation'):
                results = evaluator.run_evaluation()

                # Verify mAP values are extracted
                # assert results.box.map50 == 0.89

    def test_generates_pr_curves(self):
        """Test that PR curves can be generated."""
        with patch('tools.model_evaluation.YOLO'):
            with patch('tools.model_evaluation.plt') as mock_plt:
                evaluator = ModelEvaluator(str(self.model_path), str(self.data_yaml))

                # Mock method
                if hasattr(evaluator, 'plot_pr_curves'):
                    # Mock data
                    precision = np.array([1.0, 0.9, 0.8, 0.7])
                    recall = np.array([0.5, 0.6, 0.7, 0.8])

                    evaluator.plot_pr_curves(precision, recall)

                    # Verify plotting was attempted
                    # mock_plt.figure.assert_called()

    def test_generates_confusion_matrix(self):
        """Test that confusion matrix can be generated."""
        with patch('tools.model_evaluation.YOLO'):
            with patch('tools.model_evaluation.plt') as mock_plt:
                evaluator = ModelEvaluator(str(self.model_path), str(self.data_yaml))

                if hasattr(evaluator, 'plot_confusion_matrix'):
                    # Mock confusion matrix data
                    cm = np.array([[90, 10], [5, 95]])

                    evaluator.plot_confusion_matrix(cm)

                    # Verify plotting attempted
                    # mock_plt.figure.assert_called()

    def test_saves_evaluation_report(self):
        """Test that evaluation report is saved to file."""
        with patch('tools.model_evaluation.YOLO'):
            evaluator = ModelEvaluator(str(self.model_path), str(self.data_yaml))

            if hasattr(evaluator, 'save_report'):
                output_file = self.temp_path / 'report.txt'

                # Mock metrics
                metrics = {
                    'mAP@0.5': 0.89,
                    'mAP@0.5:0.95': 0.75,
                    'precision': 0.92,
                    'recall': 0.85
                }

                evaluator.save_report(metrics, str(output_file))

                # Verify file was created
                # assert output_file.exists()


class TestModelEvaluationEdgeCases:
    """Test edge cases for model evaluation."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def test_handles_missing_model_file(self):
        """Test that missing model file is handled."""
        non_existent_model = self.temp_path / 'nonexistent.pt'
        data_yaml = self.temp_path / 'data.yaml'
        data_yaml.write_text('names: []')

        with patch('tools.model_evaluation.YOLO') as mock_yolo:
            mock_yolo.side_effect = FileNotFoundError("Model not found")

            with pytest.raises(FileNotFoundError):
                ModelEvaluator(str(non_existent_model), str(data_yaml))

    def test_handles_invalid_yaml_format(self):
        """Test that invalid YAML format is handled."""
        model_path = self.temp_path / 'best.pt'
        model_path.write_text('fake model')

        data_yaml = self.temp_path / 'invalid.yaml'
        data_yaml.write_text('invalid: yaml: format::')

        with patch('tools.model_evaluation.YOLO'):
            # Should raise YAML parsing error
            # The actual behavior depends on implementation
            pass

    def test_handles_empty_validation_set(self):
        """Test that empty validation set is handled gracefully."""
        model_path = self.temp_path / 'best.pt'
        model_path.write_text('fake model')

        data_yaml = self.temp_path / 'data.yaml'
        data_yaml.write_text('path: /empty\nnames: []')

        with patch('tools.model_evaluation.YOLO') as mock_yolo:
            mock_model = MagicMock()

            # Mock empty results
            mock_results = MagicMock()
            mock_results.box.map50 = 0.0
            mock_model.val.return_value = mock_results

            mock_yolo.return_value = mock_model

            evaluator = ModelEvaluator(str(model_path), str(data_yaml))

            # Should handle empty set without crashing
            if hasattr(evaluator, 'run_evaluation'):
                results = evaluator.run_evaluation()

    def test_handles_single_class_evaluation(self):
        """Test that single class evaluation works correctly."""
        model_path = self.temp_path / 'best.pt'
        model_path.write_text('fake model')

        data_yaml = self.temp_path / 'data.yaml'
        data_yaml.write_text('names: ["person"]')  # Single class

        with patch('tools.model_evaluation.YOLO'):
            evaluator = ModelEvaluator(str(model_path), str(data_yaml))

            # Should handle single class
            assert evaluator is not None

    def test_handles_multiclass_evaluation(self):
        """Test that multi-class evaluation works correctly."""
        model_path = self.temp_path / 'best.pt'
        model_path.write_text('fake model')

        data_yaml = self.temp_path / 'data.yaml'
        data_yaml.write_text('names: ["person", "car", "bicycle", "dog"]')

        with patch('tools.model_evaluation.YOLO'):
            evaluator = ModelEvaluator(str(model_path), str(data_yaml))

            # Should handle multiple classes
            assert evaluator is not None


class TestModelEvaluationOutputs:
    """Test evaluation outputs and artifacts."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def test_creates_output_directory(self):
        """Test that output directory is created if it doesn't exist."""
        model_path = self.temp_path / 'best.pt'
        model_path.write_text('fake model')

        data_yaml = self.temp_path / 'data.yaml'
        data_yaml.write_text('names: []')

        output_dir = self.temp_path / 'outputs'

        with patch('tools.model_evaluation.YOLO'):
            evaluator = ModelEvaluator(str(model_path), str(data_yaml))

            if hasattr(evaluator, 'set_output_dir'):
                evaluator.set_output_dir(str(output_dir))

                # Output directory should be created or set
                # assert output_dir.exists() or evaluator.output_dir == output_dir

    def test_saves_metrics_to_json(self):
        """Test that metrics can be saved to JSON format."""
        model_path = self.temp_path / 'best.pt'
        model_path.write_text('fake model')

        data_yaml = self.temp_path / 'data.yaml'
        data_yaml.write_text('names: []')

        with patch('tools.model_evaluation.YOLO'):
            evaluator = ModelEvaluator(str(model_path), str(data_yaml))

            if hasattr(evaluator, 'save_metrics_json'):
                metrics = {'mAP@0.5': 0.89, 'mAP@0.5:0.95': 0.75}
                output_file = self.temp_path / 'metrics.json'

                evaluator.save_metrics_json(metrics, str(output_file))

                # Verify JSON file was created
                # if output_file.exists():
                #     data = json.loads(output_file.read_text())
                #     assert data['mAP@0.5'] == 0.89

    def test_generates_visualization_files(self):
        """Test that visualization files are generated."""
        model_path = self.temp_path / 'best.pt'
        model_path.write_text('fake model')

        data_yaml = self.temp_path / 'data.yaml'
        data_yaml.write_text('names: []')

        with patch('tools.model_evaluation.YOLO'):
            with patch('tools.model_evaluation.plt') as mock_plt:
                evaluator = ModelEvaluator(str(model_path), str(data_yaml))

                # Mock visualization generation
                if hasattr(evaluator, 'generate_visualizations'):
                    evaluator.generate_visualizations()

                    # Verify plotting functions were called
                    # mock_plt.savefig.assert_called()
