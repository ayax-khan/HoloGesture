import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from app.gesture_engine import GestureEngine, Gesture


def _make_landmarks(hand_type: str = "open"):
    lms = np.zeros((21, 3), dtype=np.float32)
    wrist = np.array([0.5, 0.5, 0.0])
    lms[0] = wrist

    if hand_type == "open":
        for i in range(1, 21):
            lms[i] = wrist + np.array([(i % 4) * 0.03 - 0.06, -(i // 4) * 0.04, 0.0])
        lms[4] = wrist + np.array([-0.10, -0.16, 0.0])
        lms[8] = wrist + np.array([-0.04, -0.18, 0.0])
        lms[12] = wrist + np.array([0.02, -0.18, 0.0])
        lms[16] = wrist + np.array([0.08, -0.16, 0.0])
        lms[20] = wrist + np.array([0.12, -0.10, 0.0])
    elif hand_type == "pinch":
        for i in range(1, 21):
            lms[i] = wrist + np.array([(i % 4) * 0.02 - 0.04, -(i // 4) * 0.02, 0.0])
        lms[4] = wrist + np.array([-0.03, -0.08, 0.0])
        lms[8] = wrist + np.array([-0.01, -0.08, 0.0])
        lms[12] = wrist + np.array([0.02, -0.06, 0.0])
        lms[16] = wrist + np.array([0.04, -0.04, 0.0])
        lms[20] = wrist + np.array([0.05, -0.02, 0.0])
    elif hand_type == "fist":
        for i in range(1, 21):
            lms[i] = wrist + np.array([(i % 4) * 0.015 - 0.03, -(i // 4) * 0.015, 0.0])
        for tip in [4, 8, 12, 16, 20]:
            lms[tip] = wrist + np.array([0.0, -0.02, 0.0])
    elif hand_type == "victory":
        for i in range(1, 21):
            lms[i] = wrist + np.array([(i % 4) * 0.02 - 0.04, -(i // 4) * 0.025, 0.0])
        lms[8] = wrist + np.array([-0.02, -0.14, 0.0])
        lms[12] = wrist + np.array([0.02, -0.13, 0.0])
        lms[16] = wrist + np.array([0.04, -0.03, 0.0])
        lms[20] = wrist + np.array([0.05, -0.02, 0.0])
        lms[4] = wrist + np.array([-0.04, -0.04, 0.0])
    elif hand_type == "relaxed":
        for i in range(1, 21):
            lms[i] = wrist + np.array([(i % 4) * 0.025 - 0.05, -(i // 4) * 0.025, 0.0])
        lms[4] = wrist + np.array([-0.08, -0.06, 0.0])
        lms[8] = wrist + np.array([-0.03, -0.08, 0.0])
        lms[12] = wrist + np.array([0.01, -0.08, 0.0])
        lms[16] = wrist + np.array([0.05, -0.06, 0.0])
        lms[20] = wrist + np.array([0.07, -0.03, 0.0])

    return lms


class TestGestureEngine:
    def setup_method(self):
        self.engine = GestureEngine()

    def test_none_gesture(self):
        event = self.engine.classify(_make_landmarks("open"), 0.016)
        assert event.gesture == Gesture.NONE

    def test_pinch_gesture(self):
        event = self.engine.classify(_make_landmarks("pinch"), 0.016)
        assert event.gesture == Gesture.PINCH

    def test_gesture_cooldown(self):
        self.engine.classify(_make_landmarks("pinch"), 0.016)
        event = self.engine.classify(_make_landmarks("relaxed"), 0.016)
        assert event.gesture == Gesture.PINCH

    def test_confidence_threshold(self):
        event = self.engine.classify(_make_landmarks("open"), 0.016)
        assert event.confidence >= 0.0
        assert event.confidence <= 1.0

    def test_reset(self):
        self.engine.classify(_make_landmarks("pinch"), 0.016)
        self.engine.reset()
        event = self.engine.classify(_make_landmarks("relaxed"), 0.016)
        assert event.gesture == Gesture.NONE


class TestGestureMath:
    def test_victory_angles(self):
        lms = _make_landmarks("victory")
        tip_idx = 8
        tip_mid = 12
        mcp_idx = 5
        mcp_mid = 9
        vec1 = lms[tip_idx] - lms[mcp_idx]
        vec2 = lms[tip_mid] - lms[mcp_mid]
        from app.math_utils import angle_between_vectors
        angle = angle_between_vectors(vec1, vec2)
        assert 20.0 < angle < 70.0

    def test_euclidean_distance(self):
        from app.math_utils import euclidean_distance
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([3.0, 4.0, 0.0])
        assert abs(euclidean_distance(a, b) - 5.0) < 0.001
