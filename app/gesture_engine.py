import logging
from enum import Enum, auto
from typing import Optional, List, Tuple

import numpy as np

from config import config
from app.math_utils import (
    euclidean_distance, angle_between_vectors, low_pass_filter
)

logger = logging.getLogger(__name__)


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

    def _detect_pinch(self, lms: np.ndarray) -> Tuple[bool, float]:
        dist = euclidean_distance(lms[self.THUMB_TIP], lms[self.INDEX_TIP])
        confidence = max(0.0, 1.0 - dist / (config.gesture.pinch_threshold * 3))
        return dist < config.gesture.pinch_threshold, min(confidence, 1.0)

    def _detect_open_palm(self, lms: np.ndarray) -> Tuple[bool, float]:
        tips = [self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]
        pips = [self.INDEX_PIP, self.MIDDLE_PIP, self.RING_PIP, self.PINKY_PIP]
        extended = 0
        for tip, pip in zip(tips, pips):
            angle = angle_between_vectors(lms[tip] - lms[pip], lms[pip] - lms[self.WRIST])
            if angle > config.gesture.finger_extension_angle:
                extended += 1
        thumb_angle = angle_between_vectors(
            lms[self.THUMB_TIP] - lms[self.THUMB_IP],
            lms[self.THUMB_IP] - lms[self.WRIST]
        )
        if thumb_angle > config.gesture.finger_extension_angle:
            extended += 1
        return extended >= 4, extended / 5.0

    def _detect_victory(self, lms: np.ndarray) -> Tuple[bool, float]:
        index_vec = lms[self.INDEX_TIP] - lms[self.INDEX_MCP]
        middle_vec = lms[self.MIDDLE_TIP] - lms[self.MIDDLE_MCP]
        ring_vec = lms[self.RING_TIP] - lms[self.RING_MCP]
        pinky_vec = lms[self.PINKY_TIP] - lms[self.PINKY_PIP]

        idx_mid_angle = angle_between_vectors(index_vec, middle_vec)
        ring_ext = angle_between_vectors(ring_vec, lms[self.RING_PIP] - lms[self.WRIST])
        pinky_ext = angle_between_vectors(pinky_vec, lms[self.PINKY_PIP] - lms[self.WRIST])

        index_ext = angle_between_vectors(index_vec, lms[self.INDEX_MCP] - lms[self.WRIST])
        middle_ext = angle_between_vectors(middle_vec, lms[self.MIDDLE_MCP] - lms[self.WRIST])

        condition = (
            config.gesture.victory_angle_min < idx_mid_angle < config.gesture.victory_angle_max
            and index_ext > config.gesture.finger_extension_angle
            and middle_ext > config.gesture.finger_extension_angle
            and ring_ext < config.gesture.finger_extension_angle * 0.7
            and pinky_ext < config.gesture.finger_extension_angle * 0.7
        )
        score = 0.0
        if condition:
            score = min(1.0, (idx_mid_angle - config.gesture.victory_angle_min) / 20.0)
        return condition, score

    def _detect_pointing(self, lms: np.ndarray) -> Tuple[bool, float]:
        index_vec = lms[self.INDEX_TIP] - lms[self.INDEX_MCP]
        index_ext = angle_between_vectors(index_vec, lms[self.INDEX_MCP] - lms[self.WRIST])
        folded = 0
        for tip, pip in [(self.MIDDLE_TIP, self.MIDDLE_PIP), (self.RING_TIP, self.RING_PIP), (self.PINKY_TIP, self.PINKY_PIP)]:
            angle = angle_between_vectors(lms[tip] - lms[pip], lms[pip] - lms[self.WRIST])
            if angle < config.gesture.finger_extension_angle * 0.7:
                folded += 1
        is_pointing = (
            index_ext > config.gesture.finger_extension_angle
            and folded >= 2
        )
        return is_pointing, 0.7 if is_pointing else 0.0

    def _detect_fist(self, lms: np.ndarray) -> Tuple[bool, float]:
        folded = 0
        for tip, pip in [(self.INDEX_TIP, self.INDEX_PIP), (self.MIDDLE_TIP, self.MIDDLE_PIP),
                          (self.RING_TIP, self.RING_PIP), (self.PINKY_TIP, self.PINKY_PIP)]:
            dist = euclidean_distance(lms[tip], lms[pip])
            if dist < 0.04:
                folded += 1
        thumb_over = euclidean_distance(lms[self.THUMB_TIP], lms[self.INDEX_MCP]) < 0.06
        return (folded >= 3 and thumb_over), folded / 4.0

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
        thumb_vec = lms[self.THUMB_TIP] - lms[self.THUMB_IP]
        thumb_ext = angle_between_vectors(thumb_vec, lms[self.WRIST] - lms[self.THUMB_IP])
        folded = 0
        for tip, pip in [(self.INDEX_TIP, self.INDEX_PIP), (self.MIDDLE_TIP, self.MIDDLE_PIP),
                          (self.RING_TIP, self.RING_PIP), (self.PINKY_TIP, self.PINKY_PIP)]:
            angle = angle_between_vectors(lms[tip] - lms[pip], lms[pip] - lms[self.WRIST])
            if angle < config.gesture.finger_extension_angle * 0.5:
                folded += 1
        return (thumb_ext > config.gesture.finger_extension_angle and folded >= 3), 0.75 if folded >= 3 else 0.0

    def reset(self):
        self._last_gesture = Gesture.NONE
        self._cooldown_timer = 0.0
        self._filtered_position = None
        self._last_swipe_pos = None

    @property
    def current_gesture(self) -> Gesture:
        return self._last_gesture
