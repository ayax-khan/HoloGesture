import logging
import time
from typing import Optional, List, Tuple

import numpy as np

from config import config
from app.camera import Camera
from app.hand_tracker import HandTracker, HandData
from app.gesture_engine import GestureEngine, GestureEvent, Gesture
from app.object_model import ObjectModel
from app.renderer import Renderer
from app.hud import HudOverlay
from app.state_machine import AppState, build_default_state_machine
from app.errors import CameraError, HoloError, install_global_exception_hook

logger = logging.getLogger(__name__)

GESTURE_TO_ACTION = {
    Gesture.FIST: "grab",
    Gesture.OPEN_PALM: "rotate",
    Gesture.PINCH: "scale",
    Gesture.SWIPE_LEFT: "translate",
    Gesture.SWIPE_RIGHT: "translate",
    Gesture.POINTING: "translate",
    Gesture.THUMBS_UP: "reset",
}


class ProcessingPipeline:
    def __init__(self):
        install_global_exception_hook()
        self.camera = Camera()
        self.tracker = HandTracker()
        self.left_gesture = GestureEngine()
        self.right_gesture = GestureEngine()
        self.object_model = ObjectModel()
        self.renderer = Renderer()
        self.hud = HudOverlay()
        self.state_machine = build_default_state_machine()
        self._last_frame: Optional[np.ndarray] = None
        self._hand_data_list: List[HandData] = []
        self._landmarks: Optional[list] = None
        self._last_gesture_event: Optional[GestureEvent] = None
        self._running = False
        self._fps_timer = 0.0
        self._frame_count = 0
        self._current_fps = 0
        self._last_hand_pos: Optional[np.ndarray] = None
        self._grab_last_pos: Optional[np.ndarray] = None
        self._is_grabbed: bool = False

        self._setup_state_handlers()

    def _setup_state_handlers(self):
        sm = self.state_machine
        sm.on_enter(AppState.INITIALIZING, self._on_enter_initializing)
        sm.on_exit(AppState.INITIALIZING, self._on_exit_initializing)
        sm.on_enter(AppState.ERROR, self._on_enter_error)
        sm.on_state(AppState.SEARCHING_FOR_HAND, self._on_searching)
        sm.on_state(AppState.READY, self._on_ready)
        sm.on_state(AppState.INTERACTING, self._on_interacting)
        sm.on_state(AppState.ERROR, self._on_error_state)
        sm.on_enter(AppState.SHUTTING_DOWN, self._on_shutting_down)

    def _on_enter_initializing(self):
        logger.info("Initializing HoloGesture...")
        self.hud.set_status("Initializing...")
        try:
            self.camera.open()
            logger.info("Camera initialized")
        except (CameraError, HoloError) as e:
            logger.error(f"Camera initialization failed: {e}")
            self.state_machine.force_set(AppState.ERROR)
            return
        self.state_machine.transition(AppState.SEARCHING_FOR_HAND)

    def _on_exit_initializing(self):
        self.hud.set_status("")

    def _on_enter_error(self):
        logger.warning("Entered error state")

    def _on_searching(self, dt: float):
        frame = self.camera.read()
        if frame is None:
            self._handle_camera_error()
            return
        self._last_frame = frame
        landmarks = self.tracker.process(frame)
        self._landmarks = landmarks
        self._hand_data_list = self.tracker.last_hand_data
        self.hud.set_hand_detected(landmarks is not None, dt)
        if landmarks is not None:
            self.state_machine.transition(AppState.READY)

    def _on_ready(self, dt: float):
        frame = self.camera.read()
        if frame is None:
            self._handle_camera_error()
            return
        self._last_frame = frame
        landmarks = self.tracker.process(frame)
        self._landmarks = landmarks
        self._hand_data_list = self.tracker.last_hand_data
        self.hud.set_hand_detected(landmarks is not None, dt)
        if landmarks is None:
            self.state_machine.transition(AppState.SEARCHING_FOR_HAND)
            return
        event = self.left_gesture.classify(landmarks, dt)
        self._last_gesture_event = event
        self.hud.update_gesture(event.gesture, event.confidence, dt)
        if event.gesture != Gesture.NONE:
            self.state_machine.transition(AppState.INTERACTING)

    def _on_error_state(self, dt: float):
        self._maybe_recover()

    def _on_interacting(self, dt: float):
        frame = self.camera.read()
        if frame is None:
            self._handle_camera_error()
            return
        self._last_frame = frame
        landmarks = self.tracker.process(frame)
        self._landmarks = landmarks
        self._hand_data_list = self.tracker.last_hand_data
        self.hud.set_hand_detected(landmarks is not None, dt)
        if landmarks is None:
            self._last_gesture_event = None
            self.state_machine.transition(AppState.SEARCHING_FOR_HAND)
            return

        left_event = None
        right_event = None
        for hd in self._hand_data_list:
            eng = self.left_gesture if hd.handedness == "Left" else self.right_gesture
            evt = eng.classify(hd.landmarks, dt)
            if hd.handedness == "Left":
                left_event = evt
            else:
                right_event = evt

        display_evt = left_event or right_event
        if display_evt:
            self._last_gesture_event = display_evt
            self.hud.update_gesture(display_evt.gesture, display_evt.confidence, dt)

        self._apply_two_hand_actions(left_event, right_event, dt)

        any_active = (left_event and left_event.gesture != Gesture.NONE) or \
                     (right_event and right_event.gesture != Gesture.NONE)
        if not any_active:
            self.state_machine.transition(AppState.READY)

    def _apply_two_hand_actions(self, left: Optional[GestureEvent],
                                 right: Optional[GestureEvent], dt: float):
        # Left hand → grab & move (translation)
        if left and left.gesture in (Gesture.FIST, Gesture.POINTING) and left.hand_position is not None:
            pos = left.hand_position
            if self._grab_last_pos is not None:
                dx = (pos[0] - self._grab_last_pos[0]) * 3.0
                dy = (pos[1] - self._grab_last_pos[1]) * 3.0
                dz = (self._grab_last_pos[2] - pos[2]) * 3.0
                self.object_model.translate_by(dx, -dy, dz)
            self._grab_last_pos = pos.copy()
            self._is_grabbed = True
            self._last_hand_pos = None
        elif left and left.gesture == Gesture.FIST:
            self._is_grabbed = True
        else:
            if left is None or left.gesture == Gesture.NONE:
                pass
            self._grab_last_pos = None
            self._is_grabbed = False

        # Right hand → rotate & scale
        if right and right.hand_position is not None:
            pos = right.hand_position
            if right.gesture == Gesture.OPEN_PALM:
                if self._last_hand_pos is not None:
                    dx = (pos[0] - self._last_hand_pos[0]) * 120.0
                    dy = (pos[1] - self._last_hand_pos[1]) * 120.0
                    self.object_model.rotate_by(dy, dx)
                self._last_hand_pos = pos.copy()
            elif right.gesture == Gesture.PINCH:
                if self._last_hand_pos is not None:
                    dz = (self._last_hand_pos[2] - pos[2]) * 3.0
                    self.object_model.scale_by(dz)
                self._last_hand_pos = pos.copy()
            elif right.gesture == Gesture.THUMBS_UP:
                self.object_model.reset()
            else:
                if right.gesture != Gesture.FIST and right.gesture != Gesture.POINTING:
                    self._last_hand_pos = None
        else:
            self._last_hand_pos = None

        if not self._is_grabbed:
            self._grab_last_pos = None

    def _handle_camera_error(self):
        self.hud.set_status("Camera error - attempting reconnect...")
        if self.camera.reconnect():
            self.hud.set_status("Camera reconnected")
        else:
            self.state_machine.transition(AppState.ERROR)

    def _maybe_recover(self):
        if self.camera.reconnect():
            self.hud.set_status("Recovered - searching for hand")
            self.state_machine.transition(AppState.SEARCHING_FOR_HAND)

    def update(self, dt: float):
        self.object_model.update(dt)
        self.state_machine.update(dt)
        self._update_fps(dt)
        self._check_performance()

    def _update_fps(self, dt: float):
        self._fps_timer += dt
        self._frame_count += 1
        if self._fps_timer >= 1.0:
            self._current_fps = self._frame_count
            self._frame_count = 0
            self._fps_timer = 0.0
            self.hud.set_fps(self._current_fps)

    def _check_performance(self):
        warning = (
            config.app.enable_performance_warning
            and self._current_fps > 0
            and self._current_fps < config.app.low_fps_threshold
        )
        self.hud.set_performance_warning(warning)

    def render(self):
        self.renderer.render(self.object_model, 1.0 / max(self._current_fps, 1))

    def resize(self, width: int, height: int):
        self.renderer.resize(width, height)

    def start(self):
        self._running = True
        self.state_machine.transition(AppState.INITIALIZING)

    def stop(self):
        self._running = False
        if self.state_machine.can_transition(AppState.SHUTTING_DOWN):
            self.state_machine.transition(AppState.SHUTTING_DOWN)

    def cleanup(self):
        self.tracker.close()
        self.renderer.cleanup()
        self.camera.release()
        logger.info("Pipeline cleaned up")

    @property
    def running(self) -> bool:
        return self._running

    @property
    def hud_data(self) -> dict:
        data = self.hud.get_hud_data()
        data["frame"] = self._last_frame
        data["landmarks"] = self._landmarks
        data["hand_data_list"] = self._hand_data_list
        data["hand_detected"] = self.tracker.hand_detected
        data["is_grabbed"] = self._is_grabbed
        return data

    @property
    def current_fps(self) -> int:
        return self._current_fps
