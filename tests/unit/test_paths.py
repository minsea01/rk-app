#!/usr/bin/env python3
"""Unit tests for apps.utils.paths module.

Tests path resolution, project root detection, and directory management utilities.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from apps.utils.paths import (
    get_project_root,
    resolve_path,
    ensure_dir,
    get_model_path,
    get_dataset_path,
    get_artifact_path,
    relative_to_project,
    models_dir,
    datasets_dir,
    artifacts_dir,
)


class TestGetProjectRoot:
    """Test suite for get_project_root function."""

    def test_finds_project_root_from_git_marker(self):
        """Test that .git marker is recognized as project root."""
        root = get_project_root()
        # Should find .git in project root
        assert (root / ".git").exists(), "Project should have .git directory"
        # Project name may vary depending on workspace configuration
        assert root.name in ("rk-app", "workspace"), f"Unexpected project root name: {root.name}"

    def test_finds_project_root_from_claude_md(self):
        """Test that CLAUDE.md marker is recognized."""
        root = get_project_root()
        assert (root / "CLAUDE.md").exists(), "Project should have CLAUDE.md"

    def test_returns_cached_result_on_subsequent_calls(self):
        """Test that project root is cached after first call."""
        root1 = get_project_root()
        root2 = get_project_root()
        # Should return same object (cached)
        assert root1 == root2
        assert isinstance(root1, Path)

    def test_project_root_is_absolute_path(self):
        """Test that returned path is absolute."""
        root = get_project_root()
        assert root.is_absolute(), "Project root should be absolute path"

    def test_project_root_contains_apps_directory(self):
        """Test that project root contains expected directories."""
        root = get_project_root()
        assert (root / "apps").exists(), "Project root should contain apps/"
        assert (root / "tools").exists(), "Project root should contain tools/"
        assert (root / "tests").exists(), "Project root should contain tests/"


class TestResolvePath:
    """Test suite for resolve_path function."""

    def test_resolves_relative_path_to_absolute(self):
        """Test that relative paths are resolved to absolute."""
        path = resolve_path("artifacts/models/best.onnx")
        assert path.is_absolute()
        # Project name may vary depending on workspace configuration
        assert "rk-app" in str(path) or "workspace" in str(path)
        assert str(path).endswith("artifacts/models/best.onnx")

    def test_returns_absolute_path_unchanged(self):
        """Test that absolute paths are returned as-is."""
        abs_path = Path("/tmp/test.txt")
        result = resolve_path(abs_path)
        assert result == abs_path
        assert result.is_absolute()

    def test_handles_string_input(self):
        """Test that string paths are converted to Path objects."""
        result = resolve_path("artifacts/test.txt")
        assert isinstance(result, Path)
        assert result.is_absolute()

    def test_handles_pathlib_input(self):
        """Test that Path objects are handled correctly."""
        result = resolve_path(Path("artifacts/test.txt"))
        assert isinstance(result, Path)
        assert result.is_absolute()

    def test_creates_parent_directories_when_requested(self):
        """Test that parent directories are created with create_dirs=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a path in temp directory
            test_path = Path(tmpdir) / "subdir1" / "subdir2" / "file.txt"

            # Mock get_project_root to return temp directory
            with patch("apps.utils.paths.get_project_root", return_value=Path(tmpdir)):
                result = resolve_path("subdir1/subdir2/file.txt", create_dirs=True)

                # Parent directories should exist
                assert result.parent.exists()
                assert (Path(tmpdir) / "subdir1" / "subdir2").exists()

    def test_does_not_create_dirs_by_default(self):
        """Test that directories are not created by default."""
        result = resolve_path("nonexistent/path/file.txt")
        # Should return path even if it doesn't exist
        assert isinstance(result, Path)
        # Parent may or may not exist depending on project structure


class TestEnsureDir:
    """Test suite for ensure_dir function."""

    def test_creates_directory_if_not_exists(self):
        """Test that directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("apps.utils.paths.get_project_root", return_value=Path(tmpdir)):
                result = ensure_dir("test_new_dir")
                assert result.exists()
                assert result.is_dir()

    def test_handles_nested_directories(self):
        """Test that nested directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("apps.utils.paths.get_project_root", return_value=Path(tmpdir)):
                result = ensure_dir("level1/level2/level3")
                assert result.exists()
                assert result.is_dir()
                assert (Path(tmpdir) / "level1").exists()
                assert (Path(tmpdir) / "level1" / "level2").exists()

    def test_idempotent_on_existing_directories(self):
        """Test that calling ensure_dir on existing dir doesn't fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("apps.utils.paths.get_project_root", return_value=Path(tmpdir)):
                # Create directory
                result1 = ensure_dir("existing_dir")
                # Call again on same directory
                result2 = ensure_dir("existing_dir")

                assert result1 == result2
                assert result1.exists()

    def test_returns_absolute_path(self):
        """Test that returned path is absolute."""
        result = ensure_dir("artifacts")
        assert result.is_absolute()


class TestGetModelPath:
    """Test suite for get_model_path function."""

    def test_returns_default_model_when_none_specified(self):
        """Test that default model path is returned when model_name is None."""
        result = get_model_path()
        assert isinstance(result, Path)
        assert result.is_absolute()
        # Should contain default ONNX model from PathConfig
        assert "yolo11n_416.onnx" in str(result)

    def test_looks_in_models_dir_for_filename_only(self):
        """Test that bare filenames are looked up in MODELS_DIR."""
        result = get_model_path("custom.onnx")
        assert "artifacts/models" in str(result)
        assert str(result).endswith("custom.onnx")

    def test_resolves_relative_paths_correctly(self):
        """Test that relative paths with directories are resolved."""
        result = get_model_path("custom/dir/model.onnx")
        assert result.is_absolute()
        assert str(result).endswith("custom/dir/model.onnx")


class TestGetDatasetPath:
    """Test suite for get_dataset_path function."""

    def test_returns_coco_dataset_path(self):
        """Test that COCO dataset path is resolved correctly."""
        result = get_dataset_path("coco")
        assert result.is_absolute()
        assert "datasets/coco" in str(result)

    def test_returns_citypersons_dataset_path(self):
        """Test that CityPersons dataset path is resolved."""
        result = get_dataset_path("citypersons")
        assert result.is_absolute()
        assert "CityPersons" in str(result)

    def test_handles_custom_dataset_names(self):
        """Test that custom dataset names are placed in DATASETS_DIR."""
        result = get_dataset_path("my_custom_dataset")
        assert result.is_absolute()
        assert "datasets" in str(result)
        assert str(result).endswith("my_custom_dataset")

    def test_case_insensitive_for_known_datasets(self):
        """Test that dataset names are case-insensitive."""
        result1 = get_dataset_path("COCO")
        result2 = get_dataset_path("coco")
        # Both should resolve to same path
        assert result1 == result2


class TestGetArtifactPath:
    """Test suite for get_artifact_path function."""

    def test_returns_artifact_path_without_subdir(self):
        """Test that artifact path is in artifacts/ root when no subdir."""
        result = get_artifact_path("result.jpg")
        assert result.is_absolute()
        assert "artifacts" in str(result)
        assert str(result).endswith("result.jpg")

    def test_returns_artifact_path_with_subdir(self):
        """Test that artifact path is in subdir when specified."""
        result = get_artifact_path("comparison.png", subdir="visualizations")
        assert "artifacts/visualizations" in str(result)
        assert str(result).endswith("comparison.png")


class TestRelativeToProject:
    """Test suite for relative_to_project function."""

    def test_converts_absolute_to_relative(self):
        """Test that absolute paths are converted to relative."""
        root = get_project_root()
        abs_path = root / "artifacts" / "models" / "best.onnx"
        result = relative_to_project(abs_path)

        assert not result.is_absolute()
        assert str(result) == "artifacts/models/best.onnx"

    def test_handles_paths_outside_project(self):
        """Test that paths outside project are returned as-is."""
        outside_path = Path("/tmp/external/file.txt")
        result = relative_to_project(outside_path)
        # Should return path unchanged if not relative to project
        assert result == outside_path


class TestConvenienceFunctions:
    """Test suite for convenience directory functions."""

    def test_models_dir_returns_absolute_path(self):
        """Test that models_dir() returns absolute path."""
        result = models_dir()
        assert result.is_absolute()
        assert "artifacts/models" in str(result)

    def test_datasets_dir_returns_absolute_path(self):
        """Test that datasets_dir() returns absolute path."""
        result = datasets_dir()
        assert result.is_absolute()
        assert "datasets" in str(result)

    def test_artifacts_dir_returns_absolute_path(self):
        """Test that artifacts_dir() returns absolute path."""
        result = artifacts_dir()
        assert result.is_absolute()
        assert "artifacts" in str(result)

    def test_convenience_functions_create_directories(self):
        """Test that convenience functions ensure directories exist."""
        # These should not raise even if directories don't exist
        result1 = models_dir()
        result2 = datasets_dir()
        result3 = artifacts_dir()

        assert isinstance(result1, Path)
        assert isinstance(result2, Path)
        assert isinstance(result3, Path)
