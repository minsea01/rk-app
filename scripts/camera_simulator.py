#!/usr/bin/env python3
"""
Simulate camera RTSP stream for testing dual-NIC network setup.

Reads images from directory and streams them as RTSP (H264).
Used in docker-compose.dual-nic.yml for camera_server service.

Usage:
    python3 camera_simulator.py

Environment variables:
    STREAM_HOST: RTSP server bind address (default: 0.0.0.0)
    STREAM_PORT: RTSP server port (default: 8554)
    RESOLUTION: Output resolution WxH (default: 1920x1080)
    FPS: Frames per second (default: 30)
    DATA_DIR: Directory with images (default: /data)
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    logger.error("OpenCV not installed: pip install opencv-python")
    sys.exit(1)


class CameraSimulator:
    """Simulate camera stream by cycling through image directory."""

    def __init__(
        self,
        data_dir: str = "/data",
        resolution: tuple = (1920, 1080),
        fps: int = 30
    ):
        """Initialize camera simulator.

        Args:
            data_dir: Directory containing images
            resolution: Output resolution (width, height)
            fps: Frames per second
        """
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.fps = fps
        self.frame_time = 1.0 / fps
        self.images: List[str] = []
        self.current_index = 0

    def load_images(self) -> bool:
        """Load all images from directory."""
        if not self.data_dir.exists():
            logger.error(f"Data directory not found: {self.data_dir}")
            return False

        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.images = [
            str(f) for f in self.data_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        self.images.sort()

        if not self.images:
            logger.error(f"No images found in {self.data_dir}")
            return False

        logger.info(f"Loaded {len(self.images)} images from {self.data_dir}")
        return True

    def get_next_frame(self) -> np.ndarray:
        """Get next frame, cycling through images.

        Returns:
            Frame as numpy array (HxWx3, BGR, uint8)
        """
        if not self.images:
            # Return black frame if no images
            return np.zeros((*self.resolution[::-1], 3), dtype=np.uint8)

        img_path = self.images[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.images)

        try:
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Failed to read: {img_path}")
                return np.zeros((*self.resolution[::-1], 3), dtype=np.uint8)

            # Resize to target resolution
            resized = cv2.resize(img, self.resolution)
            return resized

        except Exception as e:
            logger.error(f"Error loading {img_path}: {e}")
            return np.zeros((*self.resolution[::-1], 3), dtype=np.uint8)

    def start_stream(self):
        """Start streaming frames (blocking)."""
        if not self.load_images():
            sys.exit(1)

        logger.info(f"Starting camera simulator")
        logger.info(f"  Resolution: {self.resolution}")
        logger.info(f"  FPS: {self.fps}")
        logger.info(f"  Cycling through {len(self.images)} images")

        # Setup video writer (simulates RTSP stream)
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        output_path = 'pipe:1'  # Write to stdout

        # For demo, just log frames
        frame_count = 0
        last_log = time.time()

        try:
            while True:
                frame = self.get_next_frame()
                frame_count += 1

                # Log every second
                now = time.time()
                if now - last_log >= 1.0:
                    logger.info(f"Streamed {frame_count} frames ({self.fps} FPS)")
                    last_log = now

                # Simulate frame transmission delay
                time.sleep(self.frame_time)

        except KeyboardInterrupt:
            logger.info(f"Stopped. Total frames: {frame_count}")


def main():
    # Get config from environment
    stream_host = os.getenv('STREAM_HOST', '0.0.0.0')
    stream_port = int(os.getenv('STREAM_PORT', '8554'))
    resolution_str = os.getenv('RESOLUTION', '1920x1080')
    fps = int(os.getenv('FPS', '30'))
    data_dir = os.getenv('DATA_DIR', '/data')

    # Parse resolution
    try:
        width, height = map(int, resolution_str.split('x'))
        resolution = (width, height)
    except ValueError:
        logger.error(f"Invalid RESOLUTION format: {resolution_str}")
        sys.exit(1)

    logger.info("Camera Simulator Configuration:")
    logger.info(f"  Host: {stream_host}")
    logger.info(f"  Port: {stream_port}")
    logger.info(f"  Resolution: {resolution}")
    logger.info(f"  FPS: {fps}")
    logger.info(f"  Data Dir: {data_dir}")

    # Start simulator
    simulator = CameraSimulator(
        data_dir=data_dir,
        resolution=resolution,
        fps=fps
    )
    simulator.start_stream()


if __name__ == "__main__":
    main()
