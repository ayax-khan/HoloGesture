import logging
from typing import Optional, List

from config import config
from app.gesture_engine import Gesture

logger = logging.getLogger(__name__)


GESTURE_LABELS = {
    Gesture.NONE: "",
    Gesture.PINCH: "Pinch",
    Gesture.OPEN_PALM: "Open Palm",
    Gesture.VICTORY: "Victory",
    Gesture.POINTING: "Pointing",
    Gesture.FIST: "Fist",
    Gesture.SWIPE_LEFT: "Swipe Left",
    Gesture.SWIPE_RIGHT: "Swipe Right",
    Gesture.THUMBS_UP: "Thumbs Up",
    Gesture.SPREAD_HAND: "Spread Hand",
}

GESTURE_ACTIONS = {
    Gesture.PINCH: "Scaling object",
    Gesture.OPEN_PALM: "Rotating object",
    Gesture.VICTORY: "Victory detected",
    Gesture.POINTING: "Pointing mode",
    Gesture.FIST: "Fist detected",
    Gesture.SWIPE_LEFT: "Navigating left",
    Gesture.SWIPE_RIGHT: "Navigating right",
    Gesture.THUMBS_UP: "Confirm action",
    Gesture.SPREAD_HAND: "Spread detected",
}


class HudOverlay:
    def __init__(self):
        self._gesture_name = ""
        self._gesture_action = ""
        self._confidence = 0.0
        self._display_timer = 0.0
        self._hand_detected = False
        self._fps = 0
        self._status_message = ""
        self._performance_warning = False
        self._tracking_lost_time = 0.0

    def update_gesture(self, gesture: Gesture, confidence: float, dt: float):
        if gesture != Gesture.NONE:
            self._gesture_name = GESTURE_LABELS.get(gesture, gesture.name)
            self._gesture_action = GESTURE_ACTIONS.get(gesture, "")
            self._confidence = confidence
            self._display_timer = config.hud.gesture_display_duration
        else:
            self._display_timer = max(0.0, self._display_timer - dt)

    def set_hand_detected(self, detected: bool, dt: float):
        self._hand_detected = detected
        if not detected:
            self._tracking_lost_time += dt
        else:
            self._tracking_lost_time = 0.0

    def set_fps(self, fps: int):
        self._fps = fps

    def set_status(self, message: str):
        self._status_message = message

    def set_performance_warning(self, warning: bool):
        self._performance_warning = warning

    @property
    def tracking_lost_time(self) -> float:
        return self._tracking_lost_time

    @property
    def display_gesture(self) -> str:
        if self._display_timer > 0:
            return self._gesture_name
        return ""

    @property
    def display_action(self) -> str:
        if self._display_timer > 0:
            return self._gesture_action
        return ""

    @property
    def display_confidence(self) -> str:
        if self._display_timer > 0:
            return f"{self._confidence * 100:.0f}%"
        return ""

    @property
    def hand_status_text(self) -> str:
        if not self._hand_detected:
            elapsed = int(self._tracking_lost_time)
            if elapsed > 5:
                return "No hand detected - Please position your hand in frame"
            return "Searching for hand..."
        return "Hand detected"

    def get_hud_data(self) -> dict:
        return {
            "gesture": self.display_gesture,
            "action": self.display_action,
            "confidence": self.display_confidence,
            "hand_status": self.hand_status_text,
            "fps": self._fps,
            "performance_warning": self._performance_warning,
            "tracking_lost": not self._hand_detected,
        }
