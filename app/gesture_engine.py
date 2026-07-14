import logging
from enum import Enum, auto
from typing import Optional, List, Tuple

import numpy as np

from config import config
from app.math_utils import (
    euclidean_distance, angle_between_vectors, low_pass_filter
)

logger = logging.getLogger(__name__)

# Angle thresholds for finger states
# Angle between (TIP-PIP) and (PIP-WRIST):
#   Extended: ~0-50°  (fingertip away from wrist)
#   Folded:   ~130-180° (fingertip curled back toward wrist)
FINGER_EXTENDED_ANGLE = 50.0
FINGER_FOLDED_ANGLE = 130.0


class Gesture(Enum):
    NONE = auto()
    PINCH = auto()
    OPEN_PALM = auto()
    VICTORY = auto()
    POINTING = auto()
    FIST = auto()
    SWIPE_LEFT = auto()
    SWIPE_RIGHT = auto()
    THUMBS_UP = auto()
    SPREAD_HAND = auto()


class GestureEvent:
    def __init__(self, gesture: Gesture, confidence: float, hand_position: Optional[np.ndarray] = None):
        self.gesture = gesture
        self.confidence = confidence
        self.hand_position = hand_position


class GestureEngine:
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    THUMB_IP = 3
    THUMB_MCP = 2
    THUMB_CMC = 1
    INDEX_PIP = 6
    MIDDLE_PIP = 10
    RING_PIP = 14
    PINKY_PIP = 18
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    RING_MCP = 13
    PINKY_MCP = 17
    WRIST = 0

    HISTORY_LENGTH = 5

    def __init__(self):
        self._last_gesture = Gesture.NONE
        self._cooldown_timer = 0.0
        self._smoothed_gesture = Gesture.NONE
        self._filtered_position: Optional[np.ndarray] = None
        self._last_swipe_pos: Optional[float] = None
        self._history: List[Gesture] = []

    def classify(self, landmarks: List[np.ndarray], dt: float) -> GestureEvent:
        if landmarks is None or len(landmarks) < 21:
            return GestureEvent(Gesture.NONE, 0.0)

        lms = np.array(landmarks)
        self._cooldown_timer = max(0.0, self._cooldown_timer - dt)

        is_pinch, pinch_conf = self._detect_pinch(lms)
        is_open, open_conf = self._detect_open_palm(lms)
        is_victory, victory_conf = self._detect_victory(lms)
        is_pointing, point_conf = self._detect_pointing(lms)
        is_fist, fist_conf = self._detect_fist(lms)
        is_swipe, swipe_dir, swipe_conf = self._detect_swipe(lms)
        is_thumbs_up, thumbs_conf = self._detect_thumbs_up(lms)

        candidates = []
        if is_pinch:
            candidates.append((Gesture.PINCH, pinch_conf))
        if is_open:
            candidates.append((Gesture.OPEN_PALM, open_conf))
        if is_victory:
            candidates.append((Gesture.VICTORY, victory_conf))
        if is_pointing:
            candidates.append((Gesture.POINTING, point_conf))
        if is_fist:
            candidates.append((Gesture.FIST, fist_conf))
        if is_thumbs_up:
            candidates.append((Gesture.THUMBS_UP, thumbs_conf))
        if is_swipe:
            candidates.append((swipe_dir, swipe_conf))

        if not candidates:
            gesture = Gesture.NONE
            confidence = 0.0
        else:
            best = max(candidates, key=lambda x: x[1])
            gesture, confidence = best

        if confidence < config.gesture.confidence_threshold:
            if self._cooldown_timer > 0:
                gesture = self._last_gesture
                confidence = 0.5
            else:
                gesture = Gesture.NONE

        if gesture != Gesture.NONE:
            self._last_gesture = gesture
            self._cooldown_timer = config.gesture.gesture_cooldown

        self._history.append(gesture)
        if len(self._history) > self.HISTORY_LENGTH:
            self._history.pop(0)

        stable = self._resolve_history()
        pos = self._filter_position(lms[self.WRIST])
        return GestureEvent(stable, confidence, pos)

    def _resolve_history(self) -> Gesture:
        if not self._history:
            return Gesture.NONE
        last = self._history[-1]
        if last == Gesture.NONE:
            return last
        counts = {}
        for g in self._history:
            counts[g] = counts.get(g, 0) + 1
        best = max(counts, key=counts.get)
        majority = len(self._history) // 2 + 1
        if counts.get(best, 0) >= majority:
            return best
        return last

    def _filter_position(self, pos: np.ndarray) -> np.ndarray:
        if self._filtered_position is None:
            self._filtered_position = pos.copy()
            return self._filtered_position
        alpha = config.app.smoothing_factor
        self._filtered_position = (alpha * pos + (1 - alpha) * self._filtered_position)
        return self._filtered_position

    def _finger_angle(self, lms: np.ndarray, tip: int, pip: int) -> float:
        return angle_between_vectors(lms[tip] - lms[pip], lms[pip] - lms[self.WRIST])

    def _thumb_angle(self, lms: np.ndarray) -> float:
        return angle_between_vectors(lms[self.THUMB_TIP] - lms[self.THUMB_IP],
                                     lms[self.THUMB_IP] - lms[self.WRIST])

    def _finger_extended(self, lms: np.ndarray, tip: int, pip: int) -> bool:
        return self._finger_angle(lms, tip, pip) < FINGER_EXTENDED_ANGLE

    def _finger_folded(self, lms: np.ndarray, tip: int, pip: int) -> bool:
        return self._finger_angle(lms, tip, pip) > FINGER_FOLDED_ANGLE

    def _count_extended(self, lms: np.ndarray) -> int:
        count = 0
        for tip, pip in [(self.INDEX_TIP, self.INDEX_PIP),
                          (self.MIDDLE_TIP, self.MIDDLE_PIP),
                          (self.RING_TIP, self.RING_PIP),
                          (self.PINKY_TIP, self.PINKY_PIP)]:
            if self._finger_extended(lms, tip, pip):
                count += 1
        if self._thumb_angle(lms) < FINGER_EXTENDED_ANGLE:
            count += 1
        return count

    def _count_folded(self, lms: np.ndarray) -> int:
        count = 0
        for tip, pip in [(self.INDEX_TIP, self.INDEX_PIP),
                          (self.MIDDLE_TIP, self.MIDDLE_PIP),
                          (self.RING_TIP, self.RING_PIP),
                          (self.PINKY_TIP, self.PINKY_PIP)]:
            if self._finger_folded(lms, tip, pip):
                count += 1
        if self._thumb_angle(lms) > FINGER_FOLDED_ANGLE:
            count += 1
        return count

    def _detect_pinch(self, lms: np.ndarray) -> Tuple[bool, float]:
        dist = euclidean_distance(lms[self.THUMB_TIP], lms[self.INDEX_TIP])
        threshold = config.gesture.pinch_threshold
        confidence = max(0.0, 1.0 - dist / (threshold * 3))
        return dist < threshold, min(confidence, 1.0)

    def _detect_open_palm(self, lms: np.ndarray) -> Tuple[bool, float]:
        ext = self._count_extended(lms)
        return ext >= 4, ext / 5.0

    def _detect_victory(self, lms: np.ndarray) -> Tuple[bool, float]:
        index_ext = self._finger_extended(lms, self.INDEX_TIP, self.INDEX_PIP)
        middle_ext = self._finger_extended(lms, self.MIDDLE_TIP, self.MIDDLE_PIP)
        ring_folded = self._finger_folded(lms, self.RING_TIP, self.RING_PIP)
        pinky_folded = self._finger_folded(lms, self.PINKY_TIP, self.PINKY_PIP)

        index_vec = lms[self.INDEX_TIP] - lms[self.INDEX_MCP]
        middle_vec = lms[self.MIDDLE_TIP] - lms[self.MIDDLE_MCP]
        spread = angle_between_vectors(index_vec, middle_vec)

        condition = (
            index_ext and middle_ext
            and ring_folded and pinky_folded
            and config.gesture.victory_angle_min < spread < config.gesture.victory_angle_max
        )
        score = 0.0
        if condition:
            score = min(1.0, (spread - config.gesture.victory_angle_min) / 20.0)
        return condition, score

    def _detect_pointing(self, lms: np.ndarray) -> Tuple[bool, float]:
        index_ext = self._finger_extended(lms, self.INDEX_TIP, self.INDEX_PIP)
        others_folded = 0
        for tip, pip in [(self.MIDDLE_TIP, self.MIDDLE_PIP),
                          (self.RING_TIP, self.RING_PIP),
                          (self.PINKY_TIP, self.PINKY_PIP)]:
            if self._finger_folded(lms, tip, pip):
                others_folded += 1
        is_pointing = index_ext and others_folded >= 2
        return is_pointing, 0.7 if is_pointing else 0.0

    def _detect_fist(self, lms: np.ndarray) -> Tuple[bool, float]:
        folded = self._count_folded(lms)
        thumb_cross = euclidean_distance(lms[self.THUMB_TIP], lms[self.INDEX_MCP]) < 0.06
        return (folded >= 4 and thumb_cross), folded / 5.0

    def _detect_swipe(self, lms: np.ndarray) -> Tuple[bool, Gesture, float]:
        wrist_x = lms[self.WRIST][0]
        if self._last_swipe_pos is None:
            self._last_swipe_pos = wrist_x
            return False, Gesture.NONE, 0.0
        delta = wrist_x - self._last_swipe_pos
        self._last_swipe_pos = wrist_x
        if abs(delta) > config.gesture.swipe_threshold:
            direction = Gesture.SWIPE_RIGHT if delta > 0 else Gesture.SWIPE_LEFT
            return True, direction, min(1.0, abs(delta) * 5)
        return False, Gesture.NONE, 0.0

    def _detect_thumbs_up(self, lms: np.ndarray) -> Tuple[bool, float]:
        thumb_ext = self._thumb_angle(lms) < FINGER_EXTENDED_ANGLE
        others_folded = 0
        for tip, pip in [(self.INDEX_TIP, self.INDEX_PIP),
                          (self.MIDDLE_TIP, self.MIDDLE_PIP),
                          (self.RING_TIP, self.RING_PIP),
                          (self.PINKY_TIP, self.PINKY_PIP)]:
            if self._finger_folded(lms, tip, pip):
                others_folded += 1
        return (thumb_ext and others_folded >= 3), 0.75 if others_folded >= 3 else 0.0

    def reset(self):
        self._last_gesture = Gesture.NONE
        self._cooldown_timer = 0.0
        self._filtered_position = None
        self._last_swipe_pos = None

    @property
    def current_gesture(self) -> Gesture:
        return self._last_gesture
