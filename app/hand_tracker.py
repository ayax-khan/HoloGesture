import logging
from typing import Optional, List

import cv2
import mediapipe as mp
import numpy as np

from config import config
from app.errors import TrackingError

logger = logging.getLogger(__name__)

ModelFile = type(str)

try:
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        HandLandmarkerResult,
        RunningMode,
    )
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision.core.image import Image
    _HAS_NEW_API = True
except ImportError:
    _HAS_NEW_API = False


class HandTracker:
    LANDMARK_NAMES = [
        "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
        "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
    ]

    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    WRIST = 0

    MODEL_PATH = "hand_landmarker.task"

    def __init__(self):
        self._hand_detected = False
        self._no_hand_frames = 0
        self._last_landmarks: Optional[List[np.ndarray]] = None
        self._landmarker: Optional[HandLandmarker] = None

        if not _HAS_NEW_API:
            raise TrackingError("MediaPipe new API not available")

        try:
            base_opts = BaseOptions(model_asset_path=self.MODEL_PATH)
            options = HandLandmarkerOptions(
                base_options=base_opts,
                running_mode=RunningMode.IMAGE,
                num_hands=config.tracking.max_num_hands,
                min_hand_detection_confidence=config.tracking.min_detection_confidence,
                min_tracking_confidence=config.tracking.min_tracking_confidence,
            )
            self._landmarker = HandLandmarker.create_from_options(options)
            logger.info("HandLandmarker initialized")
        except Exception as e:
            raise TrackingError(f"Failed to initialize HandLandmarker: {e}") from e

    @property
    def hand_detected(self) -> bool:
        return self._hand_detected

    @property
    def last_landmarks(self) -> Optional[List[np.ndarray]]:
        return self._last_landmarks

    def _convert_frame(self, frame: np.ndarray) -> Optional[Image]:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image(data=rgb, image_format=mp.ImageFormat.SRGB)
        except Exception:
            return None

    def _extract_landmarks(self, result: HandLandmarkerResult) -> Optional[List[np.ndarray]]:
        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            return None
        hand = result.hand_landmarks[0]
        landmarks = []
        for lm in hand:
            landmarks.append(np.array([lm.x, lm.y, lm.z], dtype=np.float32))
        return landmarks

    def process(self, frame: np.ndarray) -> Optional[List[np.ndarray]]:
        if frame is None or self._landmarker is None:
            return None

        mp_image = self._convert_frame(frame)
        if mp_image is None:
            return None

        try:
            result = self._landmarker.detect(mp_image)
            landmarks = self._extract_landmarks(result)

            if landmarks:
                self._last_landmarks = landmarks
                self._hand_detected = True
                self._no_hand_frames = 0
                return landmarks
            else:
                self._no_hand_frames += 1
                if self._no_hand_frames > 30:
                    self._hand_detected = False
                return None
        except Exception as e:
            logger.warning(f"Hand landmarker error: {e}")
            return None

    def get_fingertip_positions(self, landmarks: List[np.ndarray]) -> dict:
        return {
            "thumb": landmarks[self.THUMB_TIP],
            "index": landmarks[self.INDEX_TIP],
            "middle": landmarks[self.MIDDLE_TIP],
            "ring": landmarks[self.RING_TIP],
            "pinky": landmarks[self.PINKY_TIP],
            "wrist": landmarks[self.WRIST]
        }

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        logger.info("Hand tracker closed")
