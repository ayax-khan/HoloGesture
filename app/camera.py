import cv2
import numpy as np
import logging
from typing import Optional, Tuple

from config import config
from app.errors import CameraError

logger = logging.getLogger(__name__)


class Camera:
    def __init__(self):
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_open = False
        self._frame_count = 0
        self._fps = 0.0

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def fps(self) -> float:
        return self._fps

    def open(self, device_id: Optional[int] = None) -> None:
        dev = device_id if device_id is not None else config.camera.device_id
        logger.info(f"Opening camera device {dev}")
        self._cap = cv2.VideoCapture(dev)
        if not self._cap.isOpened():
            raise CameraError(f"Could not open camera device {dev}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
        self._cap.set(cv2.CAP_PROP_FPS, config.camera.fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera opened: {actual_w}x{actual_h} @ {config.camera.fps}fps")
        self._is_open = True

    def read(self) -> Optional[np.ndarray]:
        if not self._is_open or self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            return None
        self._frame_count += 1
        return cv2.flip(frame, 1)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False
        logger.info("Camera released")

    def reconnect(self) -> bool:
        logger.info("Attempting camera reconnect...")
        self.release()
        try:
            self.open()
            return True
        except CameraError:
            return False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
