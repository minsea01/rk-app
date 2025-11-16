#!/usr/bin/env python3
"""
Setup configuration for RK3588 Pedestrian Detection Project
North University of China - Graduation Design
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

# Development requirements
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "pylint>=2.17.0",
    "flake8>=6.0.0",
    "isort>=5.12.0",
    "mypy>=1.0.0",
]

setup(
    name="rk3588-pedestrian-detection",
    version="1.0.0",
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
        "all": requirements + dev_requirements,
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
