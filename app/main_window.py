import logging
import time
from typing import Optional

import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QOpenGLWidget, QApplication,
    QDialog, QSlider, QComboBox, QFormLayout, QDialogButtonBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QSplashScreen
)
from PyQt5.QtCore import Qt, QTimer, QRect
import cv2
from PyQt5.QtGui import (
    QPainter, QColor, QFont, QPen, QBrush, QPixmap, QFontDatabase,
    QImage
)

from OpenGL import GL

from config import config
from app.pipeline import ProcessingPipeline
from app.renderer import Renderer
from app.state_machine import AppState
from app.gesture_engine import Gesture
from app.mesh import ObjectType, OBJECT_NAMES
from app.errors import install_global_exception_hook

logger = logging.getLogger(__name__)


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

        self.show_fps = QCheckBox()
        self.show_fps.setChecked(config.hud.show_fps)
        layout.addRow("Show FPS:", self.show_fps)

        self.show_landmarks = QCheckBox()
        self.show_landmarks.setChecked(False)
        layout.addRow("Debug Landmarks:", self.show_landmarks)

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
            "show_fps": self.show_fps.isChecked(),
            "show_landmarks": self.show_landmarks.isChecked(),
            "camera_id": self.camera_id.value(),
        }


class HoloGestureWindow(QMainWindow):
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
        self._debug_landmarks = False
        self._splash_shown = False
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

        mid_area = QHBoxLayout()
        mid_area.addStretch()
        self._cam_preview_label = QLabel()
        self._cam_preview_label.setFixedSize(160, 120)
        self._cam_preview_label.setStyleSheet("border: 1px solid rgba(0, 174, 249, 0.3); background: #0A0A1A;")
        self._cam_preview_label.hide()
        mid_area.addWidget(self._cam_preview_label)
        overlay_layout.addLayout(mid_area)

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

        debug_btn = QPushButton("◎ Debug")
        debug_btn.setCheckable(True)
        debug_btn.clicked.connect(self._toggle_debug)
        bottom_bar.addWidget(debug_btn)

        exit_btn = QPushButton("✕ Exit")
        exit_btn.clicked.connect(self.close)
        bottom_bar.addWidget(exit_btn)
        overlay_layout.addLayout(bottom_bar)

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
        self._object_label.setText(OBJECT_NAMES.get(obj_type, ""))

        fps = data.get("fps", 0)
        perf = data.get("performance_warning", False)
        fc = "#FF4444" if perf else "#00AEF9"
        self._fps_label.setStyleSheet(f"font-size: 13px; color: {fc}; padding: 4px 8px;")
        self._fps_label.setText(f"FPS: {fps}")

        status = data.get("hand_status", "")
        if data.get("tracking_lost", False):
            ss = "color: #FFAA44; background: rgba(255, 170, 68, 0.08); border-color: rgba(255, 170, 68, 0.3);"
        else:
            ss = "color: #88CCFF; background: rgba(0, 174, 249, 0.08); border-color: rgba(0, 174, 249, 0.2);"
        self._status_label.setStyleSheet(f"font-size: 14px; {ss} padding: 8px 12px; border-radius: 4px;")
        self._status_label.setText(status)

    def _update_camera_preview(self):
        frame = getattr(self._pipeline, '_last_frame', None)
        if frame is not None and self._cam_preview_label.isVisible():
            h, w = frame.shape[:2]
            fx = frame.copy()
            if h > 0 and w > 0:
                from PyQt5.QtGui import QImage
                rgb = fx[..., ::-1] if fx.shape[2] == 3 else fx
                preview = fx  # BGR
                preview_resized = cv2.resize(preview, (160, 120))
                img = QImage(preview_resized.data, 160, 120, QImage.Format_RGB888).rgbSwapped()
                self._cam_preview_label.setPixmap(QPixmap.fromImage(img))

    def _switch_object(self):
        types = list(ObjectType)
        current = self._pipeline.renderer.current_object_type
        idx = (types.index(current) + 1) % len(types)
        self._pipeline.renderer.current_object_type = types[idx]

    def _show_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            s = dlg.get_settings()
            config.app.smoothing_factor = s["smoothing"]
            config.hud.show_fps = s["show_fps"]
            self._debug_landmarks = s["show_landmarks"]
            if s["camera_id"] != config.camera.device_id:
                config.camera.device_id = s["camera_id"]
                logger.info(f"Camera ID changed to {s['camera_id']}, restart recommended")

    def _toggle_debug(self, checked):
        self._debug_landmarks = checked
        self._cam_preview_label.setVisible(checked)
        logger.info(f"Debug overlay: {'on' if checked else 'off'}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._gl_widget:
            overlay = self._gl_widget.findChild(QWidget)
            if overlay:
                overlay.setGeometry(0, 0, self._gl_widget.width(), self._gl_widget.height())

    def closeEvent(self, event):
        logger.info("Application closing...")
        if self._timer:
            self._timer.stop()
        self._pipeline.stop()
        event.accept()



