#!/usr/bin/env python3
"""
Setup configuration for RK3588 Pedestrian Detection Project
North University of China - Graduation Design
"""

from pathlib import Path
from typing import Dict, List, Tuple

from setuptools import find_packages, setup

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        tomllib = None  # type: ignore

# Read long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

def load_pyproject_dependencies() -> Tuple[str, List[str], Dict[str, List[str]]]:
    """Load version/dependencies from pyproject.toml as the single source of truth."""
    if tomllib is None:  # pragma: no cover
        raise RuntimeError("No TOML parser available (tomllib/tomli missing)")
    pyproject_file = Path(__file__).parent / "pyproject.toml"
    if not pyproject_file.exists():
        raise FileNotFoundError(f"pyproject.toml not found: {pyproject_file}")

    data = tomllib.loads(pyproject_file.read_text(encoding="utf-8"))
    project = data.get("project", {})
    version = str(project.get("version", "1.0.0"))
    dependencies = list(project.get("dependencies", []))
    optional_deps = dict(project.get("optional-dependencies", {}))

    normalized_optional: Dict[str, List[str]] = {}
    for key, value in optional_deps.items():
        if isinstance(value, list):
            normalized_optional[key] = [str(item) for item in value]
    return version, [str(dep) for dep in dependencies], normalized_optional


def load_requirements_fallback() -> List[str]:
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        return []
    with requirements_file.open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


try:
    version, requirements, optional_dependencies = load_pyproject_dependencies()
except Exception:  # pragma: no cover
    version = "1.1.0"
    requirements = load_requirements_fallback()
    optional_dependencies = {
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "pylint>=2.17.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ]
    }

dev_requirements = list(optional_dependencies.get("dev", []))
board_requirements = list(optional_dependencies.get("board", []))
train_requirements = list(optional_dependencies.get("train", []))
all_requirements = list(
    dict.fromkeys(requirements + dev_requirements + board_requirements + train_requirements)
)

setup(
    name="rk3588-pedestrian-detection",
    version=version,
    author="North University of China",
    author_email="",
    description="RK3588 Pedestrian Detection Module with YOLO and RKNN NPU Acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/minsea01/rk-app",
    packages=find_packages(include=["apps", "apps.*", "tools"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Embedded Systems",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "board": board_requirements,
        "train": train_requirements,
        "all": all_requirements,
    },
    entry_points={
        "console_scripts": [
            "rk-infer=apps.yolov8_rknn_infer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "apps": ["*.yaml", "*.yml"],
        "tools": ["*.sh"],
    },
    zip_safe=False,
    keywords=[
        "RK3588",
        "YOLO",
        "RKNN",
        "NPU",
        "object-detection",
        "pedestrian-detection",
        "edge-ai",
        "model-quantization",
        "embedded-systems",
    ],
    project_urls={
        "Bug Reports": "https://github.com/minsea01/rk-app/issues",
        "Source": "https://github.com/minsea01/rk-app",
        "Documentation": "https://github.com/minsea01/rk-app/tree/main/docs",
    },
)
