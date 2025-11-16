"""
RK3588 Application Package

Core modules for YOLO inference on RK3588 NPU.
"""

__version__ = "1.0.0"
__author__ = "North University of China"

# Core modules
from . import config
from . import exceptions
from . import logger

__all__ = ["config", "exceptions", "logger"]
