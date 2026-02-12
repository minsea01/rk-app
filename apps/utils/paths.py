#!/usr/bin/env python3
"""Path utilities for consistent path resolution across the project.

This module provides utilities to resolve relative paths to absolute paths
and ensure consistent path handling throughout the codebase.

Usage:
    from apps.utils.paths import get_project_root, resolve_path, ensure_dir

    # Get project root
    root = get_project_root()  # /home/user/rk-app

    # Resolve relative path from PathConfig
    model_path = resolve_path(PathConfig.DEFAULT_ONNX_MODEL)
    # → /home/user/rk-app/artifacts/models/yolo11n_416.onnx

    # Ensure directory exists
    ensure_dir(PathConfig.MODELS_DIR)  # Creates artifacts/models if needed
"""

import os
from pathlib import Path
from typing import Union

from apps.config import PathConfig

# Cache project root to avoid repeated lookups
_project_root = None


def get_project_root() -> Path:
    """Get absolute path to project root directory.

    Finds project root by looking for marker files (.git, CLAUDE.md, etc.)

    Returns:
        Path: Absolute path to project root

    Examples:
        >>> root = get_project_root()
        >>> root
        PosixPath('/home/user/rk-app')
    """
    global _project_root

    if _project_root is not None:
        return _project_root

    # Strategy 1: Use __file__ and walk up to find .git
    current = Path(__file__).resolve().parent
    while current != current.parent:  # Stop at filesystem root
        # Look for project markers
        if any((current / marker).exists() for marker in [".git", "CLAUDE.md", "requirements.txt"]):
            _project_root = current
            return _project_root
        current = current.parent

    # Strategy 2: Use current working directory
    cwd = Path.cwd()
    if any((cwd / marker).exists() for marker in [".git", "CLAUDE.md", "requirements.txt"]):
        _project_root = cwd
        return _project_root

    # Strategy 3: Check if we're already in project root
    # (handles case where scripts are run from project root)
    if (cwd / "apps").exists() and (cwd / "tools").exists():
        _project_root = cwd
        return _project_root

    # Fallback: Use current working directory
    _project_root = cwd
    return _project_root


def resolve_path(relative_path: Union[str, Path], create_dirs: bool = False) -> Path:
    """Resolve relative path to absolute path from project root.

    Args:
        relative_path: Relative path from project root (e.g., PathConfig.DEFAULT_ONNX_MODEL)
        create_dirs: If True, create parent directories if they don't exist

    Returns:
        Path: Absolute path

    Examples:
        >>> resolve_path('artifacts/models/best.onnx')
        PosixPath('/home/user/rk-app/artifacts/models/best.onnx')

        >>> resolve_path(PathConfig.DEFAULT_ONNX_MODEL)
        PosixPath('/home/user/rk-app/artifacts/models/yolo11n_416.onnx')

        >>> resolve_path('artifacts/new/model.onnx', create_dirs=True)
        # Creates artifacts/new/ directory
        PosixPath('/home/user/rk-app/artifacts/new/model.onnx')
    """
    path = Path(relative_path)

    # If already absolute, return as-is
    if path.is_absolute():
        return path

    # Resolve relative to project root
    absolute_path = get_project_root() / path

    # Create parent directories if requested
    if create_dirs and not absolute_path.parent.exists():
        absolute_path.parent.mkdir(parents=True, exist_ok=True)

    return absolute_path


def ensure_dir(directory: Union[str, Path]) -> Path:
    """Ensure directory exists, creating it if necessary.

    Args:
        directory: Directory path (relative or absolute)

    Returns:
        Path: Absolute path to directory

    Examples:
        >>> ensure_dir('artifacts/models')
        PosixPath('/home/user/rk-app/artifacts/models')

        >>> ensure_dir(PathConfig.VISUALIZATIONS_DIR)
        PosixPath('/home/user/rk-app/artifacts/visualizations')
    """
    dir_path = resolve_path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_model_path(model_name: str = None) -> Path:
    """Get path to model file with smart defaults.

    Args:
        model_name: Model filename or path. If None, uses DEFAULT_ONNX_MODEL.
                   If just a filename (no /), looks in MODELS_DIR.

    Returns:
        Path: Absolute path to model file

    Examples:
        >>> get_model_path()  # Uses default
        PosixPath('/home/user/rk-app/artifacts/models/yolo11n_416.onnx')

        >>> get_model_path('best.onnx')  # Looks in MODELS_DIR
        PosixPath('/home/user/rk-app/artifacts/models/best.onnx')

        >>> get_model_path('custom/dir/model.onnx')  # Uses as-is
        PosixPath('/home/user/rk-app/custom/dir/model.onnx')
    """
    if model_name is None:
        return resolve_path(PathConfig.DEFAULT_ONNX_MODEL)

    model_path = Path(model_name)

    # If it's just a filename (no directory), look in MODELS_DIR
    if len(model_path.parts) == 1:
        return resolve_path(PathConfig.MODELS_DIR) / model_name

    # Otherwise resolve as given
    return resolve_path(model_name)


def get_dataset_path(dataset_name: str = "coco") -> Path:
    """Get path to dataset directory.

    Args:
        dataset_name: Dataset name ('coco', 'coco_person', 'citypersons')

    Returns:
        Path: Absolute path to dataset directory

    Examples:
        >>> get_dataset_path('coco')
        PosixPath('/home/user/rk-app/datasets/coco')

        >>> get_dataset_path('citypersons')
        PosixPath('/home/user/rk-app/datasets/CityPersons')
    """
    dataset_map = {
        "coco": PathConfig.COCO_DIR,
        "coco_person": PathConfig.COCO_PERSON_DIR,
        "citypersons": PathConfig.CITYPERSONS_DIR,
    }

    if dataset_name.lower() in dataset_map:
        return resolve_path(dataset_map[dataset_name.lower()])
    else:
        # Assume it's a custom dataset in DATASETS_DIR
        return resolve_path(PathConfig.DATASETS_DIR) / dataset_name


def get_artifact_path(artifact_name: str, subdir: str = None) -> Path:
    """Get path to artifact file.

    Args:
        artifact_name: Artifact filename
        subdir: Subdirectory within artifacts/ (e.g., 'visualizations', 'reports')

    Returns:
        Path: Absolute path to artifact file

    Examples:
        >>> get_artifact_path('result.jpg')
        PosixPath('/home/user/rk-app/artifacts/result.jpg')

        >>> get_artifact_path('comparison.png', subdir='visualizations')
        PosixPath('/home/user/rk-app/artifacts/visualizations/comparison.png')
    """
    if subdir:
        base = resolve_path(PathConfig.ARTIFACTS_DIR) / subdir
        base.mkdir(parents=True, exist_ok=True)
        return base / artifact_name
    else:
        return resolve_path(PathConfig.ARTIFACTS_DIR) / artifact_name


def relative_to_project(absolute_path: Union[str, Path]) -> Path:
    """Convert absolute path to relative path from project root.

    Args:
        absolute_path: Absolute path

    Returns:
        Path: Relative path from project root

    Examples:
        >>> relative_to_project('/home/user/rk-app/artifacts/models/best.onnx')
        PosixPath('artifacts/models/best.onnx')
    """
    path = Path(absolute_path)
    root = get_project_root()

    try:
        return path.relative_to(root)
    except ValueError:
        # Path is not relative to project root, return as-is
        return path


# Convenience functions for common paths
def models_dir() -> Path:
    """Get models directory path."""
    return ensure_dir(PathConfig.MODELS_DIR)


def datasets_dir() -> Path:
    """Get datasets directory path."""
    return ensure_dir(PathConfig.DATASETS_DIR)


def artifacts_dir() -> Path:
    """Get artifacts directory path."""
    return ensure_dir(PathConfig.ARTIFACTS_DIR)


def logs_dir() -> Path:
    """Get logs directory path."""
    return ensure_dir(PathConfig.LOGS_DIR)


def visualizations_dir() -> Path:
    """Get visualizations directory path."""
    return ensure_dir(PathConfig.VISUALIZATIONS_DIR)
