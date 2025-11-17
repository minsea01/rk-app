#!/usr/bin/env python3
"""Headless environment detection and display fallback.

This module provides utilities to detect headless environments (no display)
and gracefully handle cv2.imshow() calls that would fail without a display.

Usage:
    from apps.utils.headless import is_headless, safe_imshow

    # Check if running headless
    if is_headless():
        print("Running in headless mode - no display available")

    # Safe imshow that automatically falls back to file saving
    safe_imshow('window_name', image, fallback_path='output.jpg')
"""
import os
import sys
from pathlib import Path
from typing import Optional

from apps.logger import setup_logger

# Lazy import cv2 to avoid hard dependency
_cv2 = None

def _get_cv2():
    """Lazy import of cv2."""
    global _cv2
    if _cv2 is None:
        try:
            import cv2
            _cv2 = cv2
        except ImportError:
            _cv2 = False  # Mark as unavailable
    return _cv2 if _cv2 is not False else None

logger = setup_logger(__name__, level='INFO')

# Cache headless detection result
_is_headless_cached = None


def is_headless() -> bool:
    """Detect if running in a headless environment (no display).

    Checks multiple indicators:
    1. DISPLAY environment variable (Linux/X11)
    2. SSH_CONNECTION environment variable (SSH session)
    3. Platform detection (some embedded systems)
    4. opencv-python vs opencv-python-headless

    Returns:
        bool: True if headless, False if display available

    Examples:
        >>> is_headless()
        False  # On desktop with display

        >>> os.environ.pop('DISPLAY', None)
        >>> is_headless()
        True  # No DISPLAY variable
    """
    global _is_headless_cached

    # Return cached result
    if _is_headless_cached is not None:
        return _is_headless_cached

    # Check 1: DISPLAY environment variable (Linux/X11)
    if 'DISPLAY' not in os.environ:
        logger.debug("Headless detected: No DISPLAY environment variable")
        _is_headless_cached = True
        return True

    # Check 2: SSH_CONNECTION indicates remote session (likely headless)
    if 'SSH_CONNECTION' in os.environ or 'SSH_CLIENT' in os.environ:
        logger.debug("Headless detected: SSH session")
        _is_headless_cached = True
        return True

    # Check 3: RK_HEADLESS environment variable (manual override)
    if os.environ.get('RK_HEADLESS', '').lower() in ('1', 'true', 'yes'):
        logger.debug("Headless detected: RK_HEADLESS environment variable")
        _is_headless_cached = True
        return True

    # Check 4: Try to detect opencv-python-headless
    cv2 = _get_cv2()
    if cv2 is not None:
        try:
            # If cv2 was built without highgui, it's headless
            if not hasattr(cv2, 'namedWindow'):
                logger.debug("Headless detected: OpenCV built without highgui")
                _is_headless_cached = True
                return True
        except Exception:
            pass

    # Check 5: Platform detection (embedded systems often headless)
    # This is a heuristic - not 100% reliable
    if sys.platform.startswith('linux'):
        # Check if running on embedded ARM board (heuristic)
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                # Common ARM SoC identifiers
                if any(marker in cpuinfo.lower() for marker in [
                    'rockchip',  # RK3588
                    'raspberry pi',
                    'jetson',
                    'armv7',
                    'armv8',
                ]):
                    # Likely embedded board - check if X server is actually running
                    if not _is_x_server_running():
                        logger.debug("Headless detected: ARM board without X server")
                        _is_headless_cached = True
                        return True
        except Exception:
            pass

    # Default: assume display is available
    logger.debug("Display detected: Headless checks passed")
    _is_headless_cached = False
    return False


def _is_x_server_running() -> bool:
    """Check if X server is actually running.

    Returns:
        bool: True if X server is running, False otherwise
    """
    try:
        # Check if DISPLAY is set and X server responds
        import subprocess
        result = subprocess.run(
            ['xdpyinfo'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def safe_imshow(
    window_name: str,
    image,
    fallback_path: Optional[str] = None,
    wait_key: int = 0
) -> bool:
    """Safe imshow that automatically handles headless environments.

    In headless mode, saves image to file instead of displaying.
    In GUI mode, displays image normally.

    Args:
        window_name: Window name for cv2.imshow()
        image: Image to display (numpy array)
        fallback_path: Path to save image in headless mode (default: auto-generated)
        wait_key: Wait key duration in ms (0 = wait forever). Ignored in headless.

    Returns:
        bool: True if displayed/saved successfully, False on error

    Examples:
        >>> # GUI mode: displays image
        >>> safe_imshow('result', image, wait_key=0)
        True

        >>> # Headless mode: saves to 'result.jpg'
        >>> safe_imshow('result', image, fallback_path='output/result.jpg')
        INFO: Headless mode - saving image to output/result.jpg
        True
    """
    cv2 = _get_cv2()
    if cv2 is None:
        # cv2 not available - must save to file
        if fallback_path is None:
            safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in window_name)
            fallback_path = f"{safe_name}.jpg"

        fallback_path = Path(fallback_path)
        fallback_path.parent.mkdir(parents=True, exist_ok=True)

        # Use PIL as fallback if available
        try:
            from PIL import Image
            if hasattr(image, 'shape'):  # numpy array
                from PIL import Image
                import numpy as np
                # Convert BGR to RGB if needed
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = image[:, :, ::-1]
                img_pil = Image.fromarray(image)
                img_pil.save(str(fallback_path))
                logger.info(f"cv2 unavailable - saved with PIL to: {fallback_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to save image (no cv2): {e}")
            return False

    if is_headless():
        # Generate fallback path if not provided
        if fallback_path is None:
            # Use window_name as filename
            safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in window_name)
            fallback_path = f"{safe_name}.jpg"

        # Ensure parent directory exists
        fallback_path = Path(fallback_path)
        fallback_path.parent.mkdir(parents=True, exist_ok=True)

        # Save image
        try:
            cv2.imwrite(str(fallback_path), image)
            logger.info(f"Headless mode - saved image to: {fallback_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save image to {fallback_path}: {e}")
            return False
    else:
        # Display normally
        try:
            cv2.imshow(window_name, image)
            if wait_key >= 0:
                cv2.waitKey(wait_key)
            return True
        except Exception as e:
            logger.error(f"Failed to display image '{window_name}': {e}")
            return False


def safe_waitKey(delay: int = 0) -> int:
    """Safe waitKey that works in headless mode.

    In headless mode, returns -1 immediately.
    In GUI mode, calls cv2.waitKey() normally.

    Args:
        delay: Wait duration in ms (0 = wait forever)

    Returns:
        int: Key code in GUI mode, -1 in headless mode
    """
    cv2 = _get_cv2()
    if cv2 is None or is_headless():
        return -1
    else:
        try:
            return cv2.waitKey(delay)
        except Exception:
            return -1


def force_headless_mode():
    """Force headless mode for testing or deployment.

    Sets RK_HEADLESS environment variable and clears cache.
    """
    global _is_headless_cached
    os.environ['RK_HEADLESS'] = '1'
    _is_headless_cached = None
    logger.info("Forced headless mode via RK_HEADLESS=1")


def force_gui_mode():
    """Force GUI mode for testing.

    Clears RK_HEADLESS environment variable and cache.
    """
    global _is_headless_cached
    os.environ.pop('RK_HEADLESS', None)
    _is_headless_cached = None
    logger.info("Forced GUI mode - cleared RK_HEADLESS")


# Auto-detect on module import
if is_headless():
    logger.info("Running in HEADLESS mode - cv2.imshow() will save to files")
else:
    logger.debug("Running in GUI mode - cv2.imshow() will display windows")
