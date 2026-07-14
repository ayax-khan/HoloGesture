import logging
import time
from typing import Optional, List

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QOpenGLWidget, QApplication,
    QDialog, QSlider, QComboBox, QFormLayout, QDialogButtonBox,
    QCheckBox, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import (
    QPainter, QColor, QFont, QPen, QBrush, QPixmap, QImage
)

from OpenGL import GL

from config import config
from app.pipeline import ProcessingPipeline
from app.renderer import Renderer
from app.state_machine import AppState
from app.gesture_engine import Gesture
from app.hand_tracker import HandData
from app.mesh import ObjectType, OBJECT_NAMES
from app.errors import install_global_exception_hook

logger = logging.getLogger(__name__)


HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


class GLWidget(QOpenGLWidget):
    def __init__(self, pipeline: ProcessingPipeline, parent=None):
        super().__init__(parent)
        self._pipeline = pipeline
        self.setMinimumSize(800, 600)

    def initializeGL(self):
        self._pipeline.renderer.initialize()
        self._pipeline.resize(self.width(), self.height())

    def paintGL(self):
        self._pipeline.render()

    def resizeGL(self, w, h):
        self._pipeline.resize(w, h)


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HoloGesture Settings")
        self.setStyleSheet("""
            QDialog { background-color: #0D0D1A; color: #00AEF9; }
            QLabel { color: #00AEF9; font-family: 'Consolas', monospace; }
            QSlider::groove:horizontal { height: 6px; background: #1A1A2E; border-radius: 3px; }
            QSlider::handle:horizontal { background: #00AEF9; width: 14px; border-radius: 7px; }
            QComboBox { background: #1A1A2E; color: #00AEF9; border: 1px solid #00AEF9; padding: 4px; }
            QSpinBox, QDoubleSpinBox { background: #1A1A2E; color: #00AEF9; border: 1px solid #00AEF9; padding: 2px; }
            QCheckBox { color: #00AEF9; }
        """)
        layout = QFormLayout(self)
        layout.setSpacing(12)

        self.sensitivity = QDoubleSpinBox()
        self.sensitivity.setRange(0.1, 5.0)
        self.sensitivity.setSingleStep(0.1)
        self.sensitivity.setValue(1.0)
        layout.addRow("Sensitivity:", self.sensitivity)

        self.smoothing = QDoubleSpinBox()
        self.smoothing.setRange(0.0, 1.0)
        self.smoothing.setSingleStep(0.05)
        self.smoothing.setValue(config.app.smoothing_factor)
        layout.addRow("Smoothing:", self.smoothing)

        self.show_landmarks = QCheckBox()
        self.show_landmarks.setChecked(True)
        layout.addRow("Show Hand Landmarks:", self.show_landmarks)

        self.camera_id = QSpinBox()
        self.camera_id.setRange(0, 10)
        self.camera_id.setValue(config.camera.device_id)
        layout.addRow("Camera ID:", self.camera_id)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.setStyleSheet("QPushButton { color: #00AEF9; border: 1px solid #00AEF9; padding: 6px 16px; }")
        layout.addRow(buttons)

    def get_settings(self) -> dict:
        return {
            "sensitivity": self.sensitivity.value(),
            "smoothing": self.smoothing.value(),
            "show_landmarks": self.show_landmarks.isChecked(),
            "camera_id": self.camera_id.value(),
        }


class HoloGestureWindow(QMainWindow):
    PREVIEW_W = 320
    PREVIEW_H = 240

    def __init__(self):
        super().__init__()
        self._pipeline = ProcessingPipeline()
        self._gl_widget: Optional[GLWidget] = None
        self._fps_label: Optional[QLabel] = None
        self._gesture_label: Optional[QLabel] = None
        self._status_label: Optional[QLabel] = None
        self._object_label: Optional[QLabel] = None
        self._cam_preview_label: Optional[QLabel] = None
        self._timer: Optional[QTimer] = None
        self._last_time = time.perf_counter()
        self._show_landmarks = True
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle(f"{config.app.app_name} v{config.app.app_version}")
        self.setMinimumSize(1024, 768)
        self.setStyleSheet("""
            QMainWindow { background-color: #0D0D1A; }
            QLabel { color: #00AEF9; font-family: 'Consolas', 'Courier New', monospace; }
            QPushButton {
                background-color: transparent; color: #00AEF9;
                border: 1px solid #00AEF9; border-radius: 4px;
                padding: 6px 16px; font-family: 'Consolas', monospace; font-size: 12px;
            }
            QPushButton:hover { background-color: #00AEF9; color: #0D0D1A; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self._gl_widget = GLWidget(self._pipeline, self)
        main_layout.addWidget(self._gl_widget, 1)

        overlay = QWidget(self._gl_widget)
        overlay.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        overlay_layout = QVBoxLayout(overlay)
        overlay_layout.setContentsMargins(12, 12, 12, 12)

        top_bar = QHBoxLayout()
        self._gesture_label = QLabel("HoloGesture")
        self._gesture_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #00E5FF;")
        top_bar.addWidget(self._gesture_label)

        self._object_label = QLabel("Cube")
        self._object_label.setStyleSheet("font-size: 13px; color: #88CCFF; padding: 4px 12px;")
        top_bar.addWidget(self._object_label)

        top_bar.addStretch()
        self._fps_label = QLabel("FPS: --")
        self._fps_label.setStyleSheet("font-size: 13px; color: #00AEF9; padding: 4px 8px;")
        top_bar.addWidget(self._fps_label)
        overlay_layout.addLayout(top_bar)

        overlay_layout.addStretch()

        bottom_bar = QHBoxLayout()
        self._status_label = QLabel("Initializing...")
        self._status_label.setStyleSheet("""
            font-size: 14px; color: #88CCFF; padding: 8px 12px;
            background: rgba(0, 174, 249, 0.08);
            border: 1px solid rgba(0, 174, 249, 0.2); border-radius: 4px;
        """)
        bottom_bar.addWidget(self._status_label)

        self._obj_btn = QPushButton("◇ Switch Object")
        self._obj_btn.clicked.connect(self._switch_object)
        bottom_bar.addWidget(self._obj_btn)

        settings_btn = QPushButton("⚙ Settings")
        settings_btn.clicked.connect(self._show_settings)
        bottom_bar.addWidget(settings_btn)

        exit_btn = QPushButton("✕ Exit")
        exit_btn.clicked.connect(self.close)
        bottom_bar.addWidget(exit_btn)
        overlay_layout.addLayout(bottom_bar)

        self._cam_preview_label = QLabel(self._gl_widget)
        self._cam_preview_label.setFixedSize(self.PREVIEW_W, self.PREVIEW_H)
        self._cam_preview_label.setStyleSheet("border: 1px solid #00AEF9; background: #0A0A1A;")
        self._cam_preview_label.move(12, 50)
        self._cam_preview_label.raise_()

        overlay.setGeometry(0, 0, self.width(), self.height())
        self._start_pipeline()

    def _start_pipeline(self):
        self._pipeline.start()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update)
        self._timer.start(16)

    def _update(self):
        now = time.perf_counter()
        dt = min(now - self._last_time, 0.05)
        self._last_time = now

        self._pipeline.update(dt)
        self._gl_widget.update()
        self._update_hud()
        self._update_camera_preview()

    def _update_hud(self):
        data = self._pipeline.hud_data
        gesture = data.get("gesture", "")
        self._gesture_label.setText(f"✦ {gesture}" if gesture else "HoloGesture")

        obj_type = self._pipeline.renderer.current_object_type
        renderer = self._pipeline.renderer
        if obj_type == ObjectType.LOADED_MODEL and renderer.has_loaded_model:
            model_name = renderer.current_model_name
            self._object_label.setText(f"📦 {model_name}")
        else:
            self._object_label.setText(OBJECT_NAMES.get(obj_type, ""))

        fps = data.get("fps", 0)
        perf = data.get("performance_warning", False)
        fc = "#FF4444" if perf else "#00AEF9"
        self._fps_label.setStyleSheet(f"font-size: 13px; color: {fc}; padding: 4px 8px;")
        self._fps_label.setText(f"FPS: {fps}")

        status = data.get("hand_status", "")
        is_grabbed = data.get("is_grabbed", False)
        hand_data_list = data.get("hand_data_list", [])
        num_hands = len([h for h in hand_data_list if h.landmarks])
        hand_info = f" [{num_hands} hand{'s' if num_hands != 1 else ''}]"

        if is_grabbed:
            status = "✊ Grabbed — move left hand to shift"
            ss = "color: #FF4444; background: rgba(255, 68, 68, 0.12); border-color: rgba(255, 68, 68, 0.4);"
        elif data.get("tracking_lost", False):
            ss = "color: #FFAA44; background: rgba(255, 170, 68, 0.08); border-color: rgba(255, 170, 68, 0.3);"
        else:
            ss = "color: #88CCFF; background: rgba(0, 174, 249, 0.08); border-color: rgba(0, 174, 249, 0.2);"
        self._status_label.setStyleSheet(f"font-size: 14px; {ss} padding: 8px 12px; border-radius: 4px;")
        self._status_label.setText(status + hand_info)

    def _draw_hand_skeleton(self, img: np.ndarray, landmarks: list,
                             lm_color, skeleton_color, thickness=2):
        for idx1, idx2 in HAND_SKELETON:
            if idx1 < len(landmarks) and idx2 < len(landmarks):
                x1 = int(landmarks[idx1][0] * self.PREVIEW_W)
                y1 = int(landmarks[idx1][1] * self.PREVIEW_H)
                x2 = int(landmarks[idx2][0] * self.PREVIEW_W)
                y2 = int(landmarks[idx2][1] * self.PREVIEW_H)
                cv2.line(img, (x1, y1), (x2, y2), skeleton_color, thickness)
        for lm in landmarks:
            x = int(lm[0] * self.PREVIEW_W)
            y = int(lm[1] * self.PREVIEW_H)
            cv2.circle(img, (x, y), 4, lm_color, -1)
            cv2.circle(img, (x, y), 4, (255, 255, 255), 1)

    def _update_camera_preview(self):
        data = self._pipeline.hud_data
        frame = data.get("frame")
        hand_data_list = data.get("hand_data_list", [])
        detected = data.get("hand_detected", False)

        if frame is None:
            return

        preview = cv2.resize(frame, (self.PREVIEW_W, self.PREVIEW_H))

        if detected and hand_data_list and self._show_landmarks:
            for hd in hand_data_list:
                if hd.handedness == "Left":
                    self._draw_hand_skeleton(
                        preview, hd.landmarks,
                        lm_color=(0, 255, 255), skeleton_color=(0, 180, 255))
                else:
                    self._draw_hand_skeleton(
                        preview, hd.landmarks,
                        lm_color=(255, 0, 255), skeleton_color=(0, 255, 128))

        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, self.PREVIEW_W, self.PREVIEW_H, QImage.Format_RGB888)
        self._cam_preview_label.setPixmap(QPixmap.fromImage(img))

    def _switch_object(self):
        renderer = self._pipeline.renderer
        current = renderer.current_object_type
        if current == ObjectType.LOADED_MODEL:
            renderer._model_manager.next()
            renderer._rebuild_loaded_mesh()
        else:
            types = [t for t in ObjectType]
            idx = (types.index(current) + 1) % len(types)
            renderer.current_object_type = types[idx]

    def _show_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            s = dlg.get_settings()
            config.app.smoothing_factor = s["smoothing"]
            self._show_landmarks = s["show_landmarks"]
            if s["camera_id"] != config.camera.device_id:
                config.camera.device_id = s["camera_id"]
                logger.info(f"Camera ID changed to {s['camera_id']}, restart recommended")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._gl_widget:
            overlay = self._gl_widget.findChild(QWidget)
            if overlay:
                overlay.setGeometry(0, 0, self._gl_widget.width(), self._gl_widget.height())
            if self._cam_preview_label:
                self._cam_preview_label.move(12, 50)

    def closeEvent(self, event):
        logger.info("Application closing...")
        if self._timer:
            self._timer.stop()
        self._pipeline.stop()
        event.accept()
