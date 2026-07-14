import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class CameraConfig:
    device_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class TrackingConfig:
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1
    max_num_hands: int = 1


@dataclass
class GestureConfig:
    pinch_threshold: float = 0.05
    finger_extension_angle: float = 140.0
    victory_angle_min: float = 25.0
    victory_angle_max: float = 65.0
    swipe_threshold: float = 0.15
    confidence_threshold: float = 0.6
    gesture_cooldown: float = 0.3


@dataclass
class RenderConfig:
    clear_color: Tuple[float, float, float, float] = (0.05, 0.05, 0.1, 1.0)
    bloom_enabled: bool = True
    glow_intensity: float = 0.6
    mesh_detail: int = 32


@dataclass
class HudConfig:
    font_size_large: int = 28
    font_size_small: int = 14
    panel_alpha: float = 0.5
    primary_color: str = "#00AEF9"
    accent_color: str = "#00E5FF"
    text_color: str = "#FFFFFF"
    bg_color: str = "#1A1A2E"
    show_fps: bool = True
    show_landmarks: bool = False
    gesture_display_duration: float = 2.0


@dataclass
class AppConfig:
    app_name: str = "HoloGesture"
    app_version: str = "1.0.0"
    target_fps: int = 60
    log_level: str = "INFO"
    log_file: str = "hologesture.log"
    enable_performance_warning: bool = True
    low_fps_threshold: int = 15
    smoothing_factor: float = 0.6
    assets_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(__file__), "assets"))
    settings_file: str = "settings.json"


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    hud: HudConfig = field(default_factory=HudConfig)
    app: AppConfig = field(default_factory=AppConfig)


config = Config()
